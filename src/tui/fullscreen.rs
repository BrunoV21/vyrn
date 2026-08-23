use super::repl::{ActiveTurnInput, Repl, ReplSnapshot, TuiUpdate, UserTurnInput};
use crate::agent::tokens::TokenCount;
use crate::llm::{ImageAttachment, LlmError};
use crate::tools::{AskUserAnswer, AskUserRequest, AskUserResponse};
use crate::vision;
use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use std::io::{IsTerminal, Stdout, stdout};
use std::time::{Duration, Instant};

const VY_BG: Color = Color::Rgb(6, 7, 10);
const VY_SURFACE: Color = Color::Rgb(13, 16, 22);
const VY_SURFACE_RAISED: Color = Color::Rgb(21, 26, 34);
const VY_BORDER: Color = Color::Rgb(39, 49, 66);
const VY_BORDER_STRONG: Color = Color::Rgb(58, 71, 94);
const VY_TEXT: Color = Color::Rgb(243, 247, 251);
const VY_MUTED: Color = Color::Rgb(152, 163, 179);
const VY_DIM: Color = Color::Rgb(103, 114, 135);
const VY_VIOLET: Color = Color::Rgb(139, 92, 246);
const VY_VIOLET_HOVER: Color = Color::Rgb(167, 139, 250);
const VY_TECH: Color = Color::Rgb(125, 162, 194);
const VY_TECH_STRONG: Color = Color::Rgb(169, 189, 211);
const VY_SUCCESS: Color = Color::Rgb(159, 232, 112);
const VY_AMBER: Color = Color::Rgb(245, 165, 36);
const VY_RED: Color = Color::Rgb(244, 63, 94);
const SPINNER_FRAME_INTERVAL: Duration = Duration::from_millis(250);

const COMMANDS: &[CommandSpec] = &[
    CommandSpec::new("/help", "list commands and keyboard controls"),
    CommandSpec::new("/stats", "provider usage, estimates, and savings"),
    CommandSpec::new("/context", "context used, available, and retained"),
    CommandSpec::new("/scratchpad", "show the last evolving turn scratchpad"),
    CommandSpec::new("/models", "switch model profile (/model alias)"),
    CommandSpec::new("/model", "alias for /models"),
    CommandSpec::new("/manifest", "show the compact machine manifest"),
    CommandSpec::new("/refresh", "rescan the machine manifest"),
    CommandSpec::new("/skills", "list discovered skill sources"),
    CommandSpec::new("/debug", "show debug trace status and path"),
    CommandSpec::new("/clear", "reset context, scratchpad, and token stats"),
    CommandSpec::new("/exit", "exit vyrn"),
];

#[derive(Debug, Clone, Copy)]
struct CommandSpec {
    name: &'static str,
    description: &'static str,
}

impl CommandSpec {
    const fn new(name: &'static str, description: &'static str) -> Self {
        Self { name, description }
    }
}

pub async fn run(repl: &mut Repl) -> anyhow::Result<()> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        anyhow::bail!("vyrn tui requires an interactive terminal");
    }

    let mut terminal = TerminalSession::enter()?;
    let mut events = EventStream::new();
    let mut ticks = tokio::time::interval(Duration::from_millis(90));
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut state = UiState::new(repl.fullscreen_snapshot());

    loop {
        terminal.draw(&mut state)?;
        let action = tokio::select! {
            event = events.next() => {
                match event {
                    Some(Ok(event)) => handle_idle_event(event, &mut state, repl),
                    Some(Err(error)) => return Err(error.into()),
                    None => return Ok(()),
                }
            }
            _ = ticks.tick() => {
                state.tick();
                IdleAction::Continue
            }
        };

        match action {
            IdleAction::Continue => {}
            IdleAction::Exit => break,
            IdleAction::Submit(input) => {
                run_turn(
                    repl,
                    &mut state,
                    &mut terminal,
                    &mut events,
                    &mut ticks,
                    input,
                )
                .await?;
            }
        }
    }

    Ok(())
}

async fn run_turn(
    repl: &mut Repl,
    state: &mut UiState,
    terminal: &mut TerminalSession,
    events: &mut EventStream,
    ticks: &mut tokio::time::Interval,
    input: UserTurnInput,
) -> anyhow::Result<()> {
    let started = Instant::now();
    repl.fullscreen_remember_prompt(&input.text);
    state.prompt_history = repl.fullscreen_snapshot().prompt_history;
    state.begin_turn(&input);

    let (active_tx, mut active_rx) = tokio::sync::mpsc::unbounded_channel();
    let (update_tx, mut update_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut deferred_actions = DeferredActiveActions::default();
    let result = {
        let turn_future = repl.handle_user_turn_fullscreen(input, &mut active_rx, move |update| {
            let _ = update_tx.send(update);
        });
        tokio::pin!(turn_future);

        loop {
            terminal.draw(state)?;
            tokio::select! {
                result = &mut turn_future => break result,
                update = update_rx.recv() => {
                    if let Some(update) = update {
                        state.apply_update(update);
                    }
                }
                event = events.next() => {
                    match event {
                        Some(Ok(event)) => {
                            handle_active_event(event, state, &active_tx, &mut deferred_actions);
                        }
                        Some(Err(error)) => {
                            let _ = active_tx.send(ActiveTurnInput::Cancel);
                            break Err(LlmError::Input(format!("terminal input failed: {error}")));
                        }
                        None => {
                            let _ = active_tx.send(ActiveTurnInput::Cancel);
                            break Err(LlmError::Canceled);
                        }
                    }
                }
                _ = ticks.tick() => state.tick(),
            }
        }
    };

    while let Ok(update) = update_rx.try_recv() {
        state.apply_update(update);
    }
    state.last_latency = started.elapsed();
    match result {
        Ok(()) => state.finish_turn(None),
        Err(LlmError::Canceled) => state.finish_turn(Some(("canceled".to_string(), false))),
        Err(error) => {
            let message = repl.fullscreen_format_error(&error);
            state.finish_turn(Some((message, true)));
        }
    }
    state.sync_snapshot(repl.fullscreen_snapshot());
    apply_deferred_active_actions(deferred_actions, state, repl);
    Ok(())
}

fn handle_idle_event(event: Event, state: &mut UiState, repl: &mut Repl) -> IdleAction {
    match event {
        Event::Resize(_, _) => IdleAction::Continue,
        Event::Paste(text) => {
            state.input.insert_str(&text);
            state.update_palette();
            IdleAction::Continue
        }
        Event::Mouse(mouse) => handle_idle_mouse(mouse, state, repl),
        Event::Key(key) if is_key_press(key) => handle_idle_key(key, state, repl),
        _ => IdleAction::Continue,
    }
}

fn handle_idle_key(key: KeyEvent, state: &mut UiState, repl: &mut Repl) -> IdleAction {
    if key.modifiers.contains(KeyModifiers::ALT) {
        state.show_shortcuts_until = Some(Instant::now() + Duration::from_millis(1400));
        return handle_alt_shortcut(key, state, repl);
    }

    if state.model_picker.is_some() {
        return handle_model_picker_key(key, state, repl);
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('c') => return IdleAction::Exit,
            KeyCode::Char('k') | KeyCode::Char('o') => {
                state.palette_open = !state.palette_open;
                state.palette_index = 0;
                return IdleAction::Continue;
            }
            KeyCode::Char('v') => {
                paste_clipboard(state);
                return IdleAction::Continue;
            }
            _ => {}
        }
    }

    match key.code {
        KeyCode::F(1) => {
            state.palette_open = true;
            state.palette_index = 0;
        }
        KeyCode::Esc => {
            if state.palette_open {
                state.palette_open = false;
            } else if state.inspector.is_some() {
                state.inspector = None;
            } else {
                return IdleAction::Exit;
            }
        }
        KeyCode::PageUp => state.scroll_up(8),
        KeyCode::PageDown => state.scroll_down(8),
        KeyCode::End => state.follow_transcript = true,
        KeyCode::Up if state.palette_open => state.palette_previous(),
        KeyCode::Down if state.palette_open => state.palette_next(),
        KeyCode::Up if !state.input.text.starts_with('/') => state.history_previous(),
        KeyCode::Down if !state.input.text.starts_with('/') => state.history_next(),
        KeyCode::Left => state.input.move_left(),
        KeyCode::Right => state.input.move_right(),
        KeyCode::Home => state.input.move_home(),
        KeyCode::Delete => {
            state.input.delete();
            state.update_palette();
        }
        KeyCode::Backspace => {
            state.input.backspace();
            state.update_palette();
        }
        KeyCode::Tab if state.palette_open => state.accept_palette(),
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
            state.input.insert_char('\n');
            state.update_palette();
        }
        KeyCode::Enter => {
            if state.palette_open {
                state.accept_palette();
            }
            return submit_idle_input(state, repl);
        }
        KeyCode::Char(ch)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::SUPER) =>
        {
            state.input.insert_char(ch);
            state.reset_history_navigation();
            state.update_palette();
        }
        _ => {}
    }
    IdleAction::Continue
}

fn handle_idle_mouse(
    mouse: crossterm::event::MouseEvent,
    state: &mut UiState,
    repl: &mut Repl,
) -> IdleAction {
    let click = register_mouse_click(mouse, state);
    let Ok((width, height)) = crossterm::terminal::size() else {
        return IdleAction::Continue;
    };
    if width < 64 || height < 18 {
        return IdleAction::Continue;
    }
    let area = Rect::new(0, 0, width, height);

    if let Some(picker) = state.model_picker.as_ref() {
        let layout = model_picker_layout(area, picker);
        if click && let Some(index) = model_picker_index_at(layout, mouse.column, mouse.row) {
            if let Some(picker) = state.model_picker.as_mut() {
                picker.selected = index;
            }
            switch_selected_model(state, repl);
            return IdleAction::Continue;
        }
        if rect_contains(layout.area, mouse.column, mouse.row) {
            if let Some(picker) = state.model_picker.as_mut() {
                match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        picker.selected = picker.selected.saturating_sub(1);
                    }
                    MouseEventKind::ScrollDown => {
                        picker.selected =
                            (picker.selected + 1).min(picker.models.len().saturating_sub(1));
                    }
                    _ => {}
                }
            }
            return IdleAction::Continue;
        }
        return IdleAction::Continue;
    }

    let chunks = interface_chunks(area, state);
    if state.palette_open {
        let matches = state.palette_matches();
        let palette = command_palette_area(chunks[5], matches.len());
        if let Some(index) =
            command_palette_index_at(palette, matches.len(), mouse.column, mouse.row)
        {
            if click {
                state.palette_index = index;
                state.accept_palette();
                return submit_idle_input(state, repl);
            }
            match mouse.kind {
                MouseEventKind::ScrollUp => {
                    state.palette_previous();
                    return IdleAction::Continue;
                }
                MouseEventKind::ScrollDown => {
                    state.palette_next();
                    return IdleAction::Continue;
                }
                _ => {}
            }
        }
    }

    if click && let Some(target) = chrome_click_target(&chunks, state, mouse.column, mouse.row) {
        return activate_click_target(target, state, repl);
    }
    if click && let Some(target) = transcript_click_target(state, mouse.column, mouse.row) {
        activate_local_click_target(target, state);
        return IdleAction::Continue;
    }

    match mouse.kind {
        MouseEventKind::ScrollUp => state.scroll_up(3),
        MouseEventKind::ScrollDown => state.scroll_down(3),
        _ => {}
    }
    IdleAction::Continue
}

fn activate_click_target(
    target: UiClickTarget,
    state: &mut UiState,
    repl: &mut Repl,
) -> IdleAction {
    if activate_local_click_target(target, state) {
        return IdleAction::Continue;
    }
    match target {
        UiClickTarget::Clear => {
            repl.fullscreen_clear();
            state.clear_session();
            state.sync_snapshot(repl.fullscreen_snapshot());
        }
        UiClickTarget::Refresh => {
            repl.fullscreen_refresh_manifest();
            state.sync_snapshot(repl.fullscreen_snapshot());
            state.open_inspector(InspectorKey::Manifest);
            state.status = "machine manifest rescanned".to_string();
        }
        UiClickTarget::OpenModels
        | UiClickTarget::ToggleTrace
        | UiClickTarget::ToggleInspect
        | UiClickTarget::Inspector(_)
        | UiClickTarget::TurnScratchpad(_)
        | UiClickTarget::ToolDetails { .. } => {
            unreachable!("local click target already handled")
        }
    }
    IdleAction::Continue
}

fn activate_local_click_target(target: UiClickTarget, state: &mut UiState) -> bool {
    state.click_feedback = Some((target, Instant::now()));
    match target {
        UiClickTarget::OpenModels => state.open_models(),
        UiClickTarget::ToggleTrace => state.trace_visible = !state.trace_visible,
        UiClickTarget::ToggleInspect => {
            state.inspect_visible = !state.inspect_visible;
            state.status = format!(
                "inspect {}",
                if state.inspect_visible {
                    "expanded"
                } else {
                    "collapsed"
                }
            );
        }
        UiClickTarget::Inspector(key) => toggle_inspector(state, key),
        UiClickTarget::TurnScratchpad(turn_index) => state.open_turn_scratchpad(turn_index),
        UiClickTarget::ToolDetails {
            turn_index,
            event_index,
        } => state.toggle_tool_details(turn_index, event_index),
        UiClickTarget::Clear | UiClickTarget::Refresh => return false,
    }
    true
}

fn register_mouse_click(mouse: crossterm::event::MouseEvent, state: &mut UiState) -> bool {
    match mouse.kind {
        MouseEventKind::Down(crossterm::event::MouseButton::Left) => {
            state.last_mouse_press = Some((mouse.column, mouse.row));
            true
        }
        MouseEventKind::Up(crossterm::event::MouseButton::Left) => {
            state.last_mouse_press.take().is_none()
        }
        _ => false,
    }
}

fn toggle_inspector(state: &mut UiState, key: InspectorKey) {
    if state.inspect_visible && state.inspector == Some(key) {
        state.inspector = None;
        state.status = format!("{} inspector collapsed", key.label());
    } else {
        state.open_inspector(key);
        state.status = format!("{} inspector expanded", key.label());
    }
}

fn rect_contains(area: Rect, column: u16, row: u16) -> bool {
    column >= area.x && column < area.right() && row >= area.y && row < area.bottom()
}

fn submit_idle_input(state: &mut UiState, repl: &mut Repl) -> IdleAction {
    let text = state.input.text.trim().to_string();
    if text.is_empty() && state.images.is_empty() {
        return IdleAction::Continue;
    }
    state.input.clear();
    state.palette_open = false;
    state.reset_history_navigation();
    if text.starts_with('/') && state.images.is_empty() {
        return run_slash_command(&text, state, repl);
    }
    let images = std::mem::take(&mut state.images);
    IdleAction::Submit(UserTurnInput { text, images })
}

fn handle_alt_shortcut(key: KeyEvent, state: &mut UiState, repl: &mut Repl) -> IdleAction {
    match key.code {
        KeyCode::Char('t') => state.trace_visible = !state.trace_visible,
        KeyCode::Char('e') => state.inspect_visible = !state.inspect_visible,
        KeyCode::Char('c') => {
            repl.fullscreen_clear();
            state.clear_session();
            state.sync_snapshot(repl.fullscreen_snapshot());
        }
        KeyCode::Char('m') => state.open_models(),
        KeyCode::Char('h') => state.open_inspector(InspectorKey::Help),
        KeyCode::Char('s') => state.open_inspector(InspectorKey::Stats),
        KeyCode::Char('k') => state.open_inspector(InspectorKey::Context),
        KeyCode::Char('p') => state.open_inspector(InspectorKey::Scratchpad),
        KeyCode::Char('i') => state.open_inspector(InspectorKey::Manifest),
        KeyCode::Char('l') => state.open_inspector(InspectorKey::Skills),
        KeyCode::Char('d') => state.open_inspector(InspectorKey::Debug),
        _ => {}
    }
    IdleAction::Continue
}

fn handle_model_picker_key(key: KeyEvent, state: &mut UiState, repl: &mut Repl) -> IdleAction {
    let Some(picker) = state.model_picker.as_mut() else {
        return IdleAction::Continue;
    };
    match key.code {
        KeyCode::Esc => state.model_picker = None,
        KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
        KeyCode::Down => {
            picker.selected = (picker.selected + 1).min(picker.models.len().saturating_sub(1));
        }
        KeyCode::Enter => switch_selected_model(state, repl),
        _ => {}
    }
    IdleAction::Continue
}

fn switch_selected_model(state: &mut UiState, repl: &mut Repl) {
    let selected = state
        .model_picker
        .as_ref()
        .and_then(|picker| picker.models.get(picker.selected))
        .cloned();
    if let Some(name) = selected
        && repl.fullscreen_switch_model(&name)
    {
        state.status = format!("model switched → {name}");
        state.model_picker = None;
        state.sync_snapshot(repl.fullscreen_snapshot());
    }
}

fn run_slash_command(input: &str, state: &mut UiState, repl: &mut Repl) -> IdleAction {
    match input.split_whitespace().next().unwrap_or_default() {
        "/exit" => return IdleAction::Exit,
        "/help" => state.open_inspector(InspectorKey::Help),
        "/stats" => state.open_inspector(InspectorKey::Stats),
        "/context" => state.open_inspector(InspectorKey::Context),
        "/scratchpad" => state.open_inspector(InspectorKey::Scratchpad),
        "/manifest" => state.open_inspector(InspectorKey::Manifest),
        "/skills" => state.open_inspector(InspectorKey::Skills),
        "/debug" => state.open_inspector(InspectorKey::Debug),
        "/models" | "/model" => state.open_models(),
        "/refresh" => {
            repl.fullscreen_refresh_manifest();
            state.sync_snapshot(repl.fullscreen_snapshot());
            state.open_inspector(InspectorKey::Manifest);
            state.status = "machine manifest rescanned".to_string();
        }
        "/clear" => {
            repl.fullscreen_clear();
            state.clear_session();
            state.sync_snapshot(repl.fullscreen_snapshot());
        }
        command => state.status = format!("unknown command {command} · try /help"),
    }
    IdleAction::Continue
}

#[derive(Debug, Default)]
struct DeferredActiveActions {
    clear: bool,
    refresh_manifest: bool,
    model: Option<String>,
}

fn apply_deferred_active_actions(
    actions: DeferredActiveActions,
    state: &mut UiState,
    repl: &mut Repl,
) {
    if actions.clear {
        repl.fullscreen_clear();
        state.clear_session();
        state.sync_snapshot(repl.fullscreen_snapshot());
    }
    if actions.refresh_manifest {
        repl.fullscreen_refresh_manifest();
        state.sync_snapshot(repl.fullscreen_snapshot());
        state.open_inspector(InspectorKey::Manifest);
        state.status = "machine manifest rescanned".to_string();
    }
    if let Some(name) = actions.model
        && repl.fullscreen_switch_model(&name)
    {
        state.status = format!("model switched → {name}");
        state.model_picker = None;
        state.sync_snapshot(repl.fullscreen_snapshot());
    }
}

fn handle_active_event(
    event: Event,
    state: &mut UiState,
    active_tx: &tokio::sync::mpsc::UnboundedSender<ActiveTurnInput>,
    deferred_actions: &mut DeferredActiveActions,
) {
    match event {
        Event::Resize(_, _) => {}
        Event::Paste(text) => state.input.insert_str(&text),
        Event::Mouse(mouse) => {
            handle_active_mouse(mouse, state, active_tx, deferred_actions);
        }
        Event::Key(key) if is_key_press(key) => {
            if state.clarification.is_some() {
                if let Some(action) = state.handle_clarification_key(key) {
                    match action {
                        ClarificationAction::Answer(response) => {
                            let _ = active_tx.send(ActiveTurnInput::Clarification(response));
                        }
                        ClarificationAction::Cancel => {
                            let _ = active_tx.send(ActiveTurnInput::Cancel);
                        }
                    }
                }
                return;
            }

            if state.model_picker.is_some() {
                handle_active_model_picker_key(key, state, deferred_actions);
                return;
            }

            if key.modifiers.contains(KeyModifiers::CONTROL)
                && matches!(key.code, KeyCode::Char('c'))
            {
                state.status = "canceling active turn…".to_string();
                let _ = active_tx.send(ActiveTurnInput::Cancel);
                return;
            }
            if key.modifiers.contains(KeyModifiers::CONTROL)
                && matches!(key.code, KeyCode::Char('v'))
            {
                paste_clipboard(state);
                return;
            }

            match key.code {
                KeyCode::Esc => {
                    state.status = "canceling active turn…".to_string();
                    let _ = active_tx.send(ActiveTurnInput::Cancel);
                }
                KeyCode::PageUp => state.scroll_up(8),
                KeyCode::PageDown => state.scroll_down(8),
                KeyCode::End => state.follow_transcript = true,
                KeyCode::Left => state.input.move_left(),
                KeyCode::Right => state.input.move_right(),
                KeyCode::Home => state.input.move_home(),
                KeyCode::Delete => state.input.delete(),
                KeyCode::Backspace => state.input.backspace(),
                KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                    state.input.insert_char('\n');
                }
                KeyCode::Enter => {
                    let text = state.input.text.trim().to_string();
                    if !text.is_empty() {
                        state.input.clear();
                        state.status = "steering queued".to_string();
                        let _ = active_tx.send(ActiveTurnInput::Steering(text));
                    }
                }
                KeyCode::Char(ch)
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::SUPER) =>
                {
                    state.input.insert_char(ch);
                }
                _ => {}
            }
        }
        _ => {}
    }
}

fn handle_active_mouse(
    mouse: crossterm::event::MouseEvent,
    state: &mut UiState,
    active_tx: &tokio::sync::mpsc::UnboundedSender<ActiveTurnInput>,
    deferred_actions: &mut DeferredActiveActions,
) {
    let Ok((width, height)) = crossterm::terminal::size() else {
        return;
    };
    handle_active_mouse_in_area(
        mouse,
        Rect::new(0, 0, width, height),
        state,
        active_tx,
        deferred_actions,
    );
}

fn handle_active_mouse_in_area(
    mouse: crossterm::event::MouseEvent,
    area: Rect,
    state: &mut UiState,
    active_tx: &tokio::sync::mpsc::UnboundedSender<ActiveTurnInput>,
    deferred_actions: &mut DeferredActiveActions,
) {
    let click = register_mouse_click(mouse, state);
    if area.width < 64 || area.height < 18 || state.clarification.is_some() {
        return;
    }

    if let Some(picker) = state.model_picker.as_ref() {
        let layout = model_picker_layout(area, picker);
        if click && let Some(index) = model_picker_index_at(layout, mouse.column, mouse.row) {
            queue_active_model_switch(index, state, deferred_actions);
            return;
        }
        if rect_contains(layout.area, mouse.column, mouse.row)
            && let Some(picker) = state.model_picker.as_mut()
        {
            match mouse.kind {
                MouseEventKind::ScrollUp => {
                    picker.selected = picker.selected.saturating_sub(1);
                }
                MouseEventKind::ScrollDown => {
                    picker.selected =
                        (picker.selected + 1).min(picker.models.len().saturating_sub(1));
                }
                _ => {}
            }
        }
        return;
    }

    let chunks = interface_chunks(area, state);
    if click && let Some(target) = chrome_click_target(&chunks, state, mouse.column, mouse.row) {
        if activate_local_click_target(target, state) {
            return;
        }
        match target {
            UiClickTarget::Clear => {
                deferred_actions.clear = true;
                state.status = "clear queued · canceling active turn…".to_string();
                let _ = active_tx.send(ActiveTurnInput::Cancel);
            }
            UiClickTarget::Refresh => {
                deferred_actions.refresh_manifest = true;
                state.status = "manifest refresh queued for after this turn".to_string();
            }
            UiClickTarget::OpenModels
            | UiClickTarget::ToggleTrace
            | UiClickTarget::ToggleInspect
            | UiClickTarget::Inspector(_)
            | UiClickTarget::TurnScratchpad(_)
            | UiClickTarget::ToolDetails { .. } => {
                unreachable!("local click target already handled")
            }
        }
        return;
    }
    if click && let Some(target) = transcript_click_target(state, mouse.column, mouse.row) {
        activate_local_click_target(target, state);
        return;
    }

    match mouse.kind {
        MouseEventKind::ScrollUp => state.scroll_up(3),
        MouseEventKind::ScrollDown => state.scroll_down(3),
        _ => {}
    }
}

fn handle_active_model_picker_key(
    key: KeyEvent,
    state: &mut UiState,
    deferred_actions: &mut DeferredActiveActions,
) {
    let Some(picker) = state.model_picker.as_mut() else {
        return;
    };
    match key.code {
        KeyCode::Esc => state.model_picker = None,
        KeyCode::Up => picker.selected = picker.selected.saturating_sub(1),
        KeyCode::Down => {
            picker.selected = (picker.selected + 1).min(picker.models.len().saturating_sub(1));
        }
        KeyCode::Enter => queue_active_model_switch(picker.selected, state, deferred_actions),
        _ => {}
    }
}

fn queue_active_model_switch(
    index: usize,
    state: &mut UiState,
    deferred_actions: &mut DeferredActiveActions,
) {
    let selected = state
        .model_picker
        .as_ref()
        .and_then(|picker| picker.models.get(index))
        .cloned();
    if let Some(name) = selected {
        deferred_actions.model = Some(name.clone());
        state.model_picker = None;
        state.status = format!("model switch queued → {name}");
    }
}

fn is_key_press(key: KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
}

fn paste_clipboard(state: &mut UiState) {
    if state.images.len() < vision::MAX_IMAGES_PER_MESSAGE {
        match vision::image_from_clipboard() {
            Ok(Some(image)) => {
                state.images.push(image);
                state.status = format!("{} image(s) attached", state.images.len());
                return;
            }
            Ok(None) => {}
            Err(error) => state.status = format!("clipboard image unavailable: {error}"),
        }
    }
    match vision::text_from_clipboard() {
        Ok(Some(text)) => {
            state.input.insert_str(&text);
            state.update_palette();
        }
        Ok(None) => {}
        Err(error) => state.status = format!("clipboard unavailable: {error}"),
    }
}

enum IdleAction {
    Continue,
    Submit(UserTurnInput),
    Exit,
}

#[derive(Debug, Clone, Default)]
struct InputBuffer {
    text: String,
    cursor: usize,
}

impl InputBuffer {
    fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    fn set(&mut self, text: String) {
        self.cursor = text.chars().count();
        self.text = text;
    }

    fn insert_char(&mut self, ch: char) {
        let byte = byte_index(&self.text, self.cursor);
        self.text.insert(byte, ch);
        self.cursor += 1;
    }

    fn insert_str(&mut self, text: &str) {
        let byte = byte_index(&self.text, self.cursor);
        self.text.insert_str(byte, text);
        self.cursor += text.chars().count();
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let start = byte_index(&self.text, self.cursor - 1);
        let end = byte_index(&self.text, self.cursor);
        self.text.replace_range(start..end, "");
        self.cursor -= 1;
    }

    fn delete(&mut self) {
        if self.cursor >= self.text.chars().count() {
            return;
        }
        let start = byte_index(&self.text, self.cursor);
        let end = byte_index(&self.text, self.cursor + 1);
        self.text.replace_range(start..end, "");
    }

    fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    fn move_right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.text.chars().count());
    }

    fn move_home(&mut self) {
        let before = self.text.chars().take(self.cursor).collect::<String>();
        self.cursor = before
            .rfind('\n')
            .map_or(0, |index| before[..=index].chars().count());
    }
}

fn byte_index(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .nth(char_index)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspectorKey {
    Help,
    Stats,
    Context,
    Scratchpad,
    Manifest,
    Skills,
    Debug,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UiClickTarget {
    OpenModels,
    ToggleTrace,
    ToggleInspect,
    Clear,
    Inspector(InspectorKey),
    TurnScratchpad(usize),
    ToolDetails {
        turn_index: usize,
        event_index: usize,
    },
    Refresh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClickRegion {
    area: Rect,
    target: UiClickTarget,
}

impl ClickRegion {
    fn contains(self, column: u16, row: u16) -> bool {
        column >= self.area.x
            && column < self.area.right()
            && row >= self.area.y
            && row < self.area.bottom()
    }
}

impl InspectorKey {
    fn label(self) -> &'static str {
        match self {
            Self::Help => "help",
            Self::Stats => "stats",
            Self::Context => "context",
            Self::Scratchpad => "scratchpad",
            Self::Manifest => "manifest",
            Self::Skills => "skills",
            Self::Debug => "debug",
        }
    }
}

#[derive(Debug)]
struct ModelPicker {
    models: Vec<String>,
    selected: usize,
}

#[derive(Debug)]
struct ClarificationUi {
    request: AskUserRequest,
    question_index: usize,
    selected: usize,
    answers: Vec<AskUserAnswer>,
    freeform: bool,
    input: InputBuffer,
}

impl ClarificationUi {
    fn new(request: AskUserRequest) -> Self {
        Self {
            request,
            question_index: 0,
            selected: 0,
            answers: Vec::new(),
            freeform: false,
            input: InputBuffer::default(),
        }
    }

    fn current_question(&self) -> &crate::tools::AskUserQuestion {
        &self.request.questions[self.question_index]
    }

    fn selection_count(&self) -> usize {
        self.current_question().options.len() + 1
    }

    fn submit(&mut self) -> Option<AskUserResponse> {
        let question = self.current_question().clone();
        let answer = if self.freeform {
            let answer = self.input.text.trim().to_string();
            if answer.is_empty() {
                return None;
            }
            AskUserAnswer::Freeform {
                id: question.id,
                answer,
            }
        } else if self.selected < question.options.len() {
            let option = &question.options[self.selected];
            AskUserAnswer::Option {
                id: question.id,
                answer: option.label.clone(),
                option_index: self.selected,
                option_label: option.label.clone(),
            }
        } else {
            self.freeform = true;
            self.input.clear();
            return None;
        };
        self.answers.push(answer);
        if self.question_index + 1 < self.request.questions.len() {
            self.question_index += 1;
            self.selected = 0;
            self.freeform = false;
            self.input.clear();
            None
        } else {
            Some(AskUserResponse {
                answers: std::mem::take(&mut self.answers),
            })
        }
    }
}

enum ClarificationAction {
    Answer(AskUserResponse),
    Cancel,
}

#[derive(Debug)]
struct UiState {
    snapshot: ReplSnapshot,
    turns: Vec<TurnView>,
    input: InputBuffer,
    images: Vec<ImageAttachment>,
    prompt_history: Vec<String>,
    history_index: Option<usize>,
    history_draft: String,
    palette_open: bool,
    palette_index: usize,
    inspect_visible: bool,
    inspector: Option<InspectorKey>,
    inspected_turn: Option<usize>,
    inspector_body: String,
    trace_visible: bool,
    model_picker: Option<ModelPicker>,
    clarification: Option<ClarificationUi>,
    running: bool,
    activity: Option<String>,
    status: String,
    spinner_frame: usize,
    last_spinner_advance: Instant,
    follow_transcript: bool,
    transcript_scroll: u16,
    transcript_max_scroll: u16,
    tool_count: usize,
    last_latency: Duration,
    show_shortcuts_until: Option<Instant>,
    last_mouse_press: Option<(u16, u16)>,
    click_feedback: Option<(UiClickTarget, Instant)>,
    transcript_click_regions: Vec<ClickRegion>,
}

impl UiState {
    fn new(snapshot: ReplSnapshot) -> Self {
        let prompt_history = snapshot.prompt_history.clone();
        let inspector_body = snapshot.scratchpad.clone();
        Self {
            snapshot,
            turns: Vec::new(),
            input: InputBuffer::default(),
            images: Vec::new(),
            prompt_history,
            history_index: None,
            history_draft: String::new(),
            palette_open: false,
            palette_index: 0,
            inspect_visible: true,
            inspector: Some(InspectorKey::Scratchpad),
            inspected_turn: None,
            inspector_body,
            trace_visible: true,
            model_picker: None,
            clarification: None,
            running: false,
            activity: None,
            status: "ready · / or Ctrl+K for commands".to_string(),
            spinner_frame: 0,
            last_spinner_advance: Instant::now(),
            follow_transcript: true,
            transcript_scroll: 0,
            transcript_max_scroll: 0,
            tool_count: 0,
            last_latency: Duration::ZERO,
            show_shortcuts_until: None,
            last_mouse_press: None,
            click_feedback: None,
            transcript_click_regions: Vec::new(),
        }
    }

    fn sync_snapshot(&mut self, snapshot: ReplSnapshot) {
        self.prompt_history = snapshot.prompt_history.clone();
        self.snapshot = snapshot;
        self.refresh_inspector_body();
    }

    fn tick(&mut self) {
        if self.last_spinner_advance.elapsed() >= SPINNER_FRAME_INTERVAL {
            self.spinner_frame = self.spinner_frame.wrapping_add(1);
            self.last_spinner_advance = Instant::now();
        }
        if self
            .show_shortcuts_until
            .is_some_and(|until| Instant::now() >= until)
        {
            self.show_shortcuts_until = None;
        }
        if self
            .click_feedback
            .is_some_and(|(_, clicked)| clicked.elapsed() >= Duration::from_millis(500))
        {
            self.click_feedback = None;
        }
    }

    fn begin_turn(&mut self, input: &UserTurnInput) {
        self.turns.push(TurnView {
            user: input.text.clone(),
            image_count: input.images.len(),
            events: Vec::new(),
            stats: None,
            scratchpad: None,
            complete: false,
        });
        self.running = true;
        self.activity = Some("routing".to_string());
        self.status = "active · type + Enter to steer · Esc to cancel".to_string();
        self.follow_transcript = true;
    }

    fn finish_turn(&mut self, outcome: Option<(String, bool)>) {
        self.running = false;
        self.activity = None;
        self.clarification = None;
        if let Some((message, is_error)) = outcome
            && let Some(turn) = self.turns.last_mut()
        {
            turn.events.push(if is_error {
                TurnEvent::Error(message)
            } else {
                TurnEvent::System(message)
            });
        }
        if let Some(turn) = self.turns.last_mut() {
            turn.complete = true;
        }
        self.status = "ready · / or Ctrl+K for commands".to_string();
        self.follow_transcript = true;
    }

    fn apply_update(&mut self, update: TuiUpdate) {
        let Some(turn) = self.turns.last_mut() else {
            return;
        };
        let mut scratchpad_changed = false;
        match update {
            TuiUpdate::SummaryStart => {
                self.activity = Some("integrating previous turn".to_string());
                push_trace(
                    turn,
                    "memory",
                    "integrating previous turn",
                    TraceState::Live,
                );
            }
            TuiUpdate::SummaryDone => finish_last_trace(turn, "memory"),
            TuiUpdate::AssistantStart => {
                self.activity = Some("thinking".to_string());
                self.spinner_frame = 0;
                self.last_spinner_advance = Instant::now();
            }
            TuiUpdate::AssistantDelta(delta) => {
                self.activity = Some("streaming response".to_string());
                match turn.events.last_mut() {
                    Some(TurnEvent::Answer(answer)) => answer.push_str(&delta),
                    _ => turn.events.push(TurnEvent::Answer(delta)),
                }
            }
            TuiUpdate::AssistantDone => self.activity = None,
            TuiUpdate::AssistantInterrupted => {
                self.activity = Some("applying live steering".to_string());
                push_trace(turn, "llm", "request interrupted", TraceState::Warning);
            }
            TuiUpdate::ToolStarted { name, input } => {
                self.activity = Some(format!("running {name}"));
                self.tool_count += 1;
                turn.events.push(TurnEvent::Tool(ToolCard {
                    name,
                    input,
                    output: String::new(),
                    state: ToolState::Running,
                    started: Instant::now(),
                    elapsed: Duration::ZERO,
                    expanded: false,
                }));
            }
            TuiUpdate::ToolInputStart => {
                self.activity = Some("waiting for clarification".to_string());
            }
            TuiUpdate::ToolOk { name, preview } => {
                self.activity = None;
                finish_tool(turn, &name, preview, ToolState::Success);
            }
            TuiUpdate::ToolError { name, error } => {
                self.activity = None;
                finish_tool(turn, &name, error, ToolState::Failure);
            }
            TuiUpdate::ScratchpadStart => {
                self.activity = Some("compacting turn memory".to_string());
                push_trace(turn, "memory", "compacting tool context", TraceState::Live);
            }
            TuiUpdate::ScratchpadDone {
                summary,
                output_tokens,
            } => {
                finish_last_trace(turn, "memory");
                if let (Some(summary), Some(tokens)) = (summary, output_tokens) {
                    turn.scratchpad = Some(ScratchpadView { summary, tokens });
                    scratchpad_changed = true;
                }
            }
            TuiUpdate::Steering(text) => {
                turn.events
                    .push(TurnEvent::System(format!("live steering · {text}")));
            }
            TuiUpdate::Stats(stats) => turn.stats = Some(stats),
            TuiUpdate::Summary(summary) => turn
                .events
                .push(TurnEvent::System(format!("summary\n{summary}"))),
            TuiUpdate::Clarification(request) => {
                self.clarification = Some(ClarificationUi::new(request));
                self.activity = Some("waiting for your answer".to_string());
            }
        }
        if scratchpad_changed {
            self.refresh_inspector_body();
        }
        self.follow_transcript = true;
    }

    fn clear_session(&mut self) {
        self.turns.clear();
        self.tool_count = 0;
        self.last_latency = Duration::ZERO;
        self.inspector = None;
        self.inspected_turn = None;
        self.model_picker = None;
        self.clarification = None;
        self.status = "session cleared".to_string();
        self.follow_transcript = true;
        self.transcript_scroll = 0;
    }

    fn open_inspector(&mut self, key: InspectorKey) {
        self.inspect_visible = true;
        self.inspector = Some(key);
        self.inspected_turn = None;
        self.refresh_inspector_body();
    }

    fn open_turn_scratchpad(&mut self, turn_index: usize) {
        if self.turns.get(turn_index).is_none() {
            return;
        }
        self.inspect_visible = true;
        self.inspector = Some(InspectorKey::Scratchpad);
        self.inspected_turn = Some(turn_index);
        self.refresh_inspector_body();
        self.status = format!("interaction {} scratchpad expanded", turn_index + 1);
    }

    fn toggle_tool_details(&mut self, turn_index: usize, event_index: usize) {
        let Some(TurnEvent::Tool(tool)) = self
            .turns
            .get_mut(turn_index)
            .and_then(|turn| turn.events.get_mut(event_index))
        else {
            return;
        };
        tool.expanded = !tool.expanded;
        self.status = format!(
            "tool.{} details {}",
            tool.name,
            if tool.expanded {
                "expanded"
            } else {
                "collapsed"
            }
        );
        self.follow_transcript = false;
    }

    fn refresh_inspector_body(&mut self) {
        let Some(key) = self.inspector else {
            return;
        };
        self.inspector_body = if key == InspectorKey::Scratchpad {
            if let Some(index) = self.inspected_turn {
                self.turns
                    .get(index)
                    .and_then(|turn| turn.scratchpad.as_ref())
                    .map(|scratchpad| turn_scratchpad_text(index, scratchpad))
                    .unwrap_or_else(|| {
                        format!(
                            "interaction {} scratchpad: none (no tool-driven context was compacted)",
                            index + 1
                        )
                    })
            } else {
                self.snapshot.scratchpad.clone()
            }
        } else {
            inspector_text(key, &self.snapshot)
        };
    }

    fn open_models(&mut self) {
        let models = self.snapshot.models.clone();
        let selected = models
            .iter()
            .position(|model| model == &self.snapshot.model_name)
            .unwrap_or_default();
        self.model_picker = Some(ModelPicker { models, selected });
    }

    fn update_palette(&mut self) {
        self.palette_open =
            self.input.text.starts_with('/') && !self.input.text.contains(char::is_whitespace);
        self.palette_index = self
            .palette_index
            .min(self.palette_matches().len().saturating_sub(1));
    }

    fn palette_matches(&self) -> Vec<CommandSpec> {
        let prefix = self.input.text.trim();
        COMMANDS
            .iter()
            .copied()
            .filter(|command| prefix.is_empty() || command.name.starts_with(prefix))
            .collect()
    }

    fn palette_previous(&mut self) {
        let len = self.palette_matches().len();
        if len > 0 {
            self.palette_index = if self.palette_index == 0 {
                len - 1
            } else {
                self.palette_index - 1
            };
        }
    }

    fn palette_next(&mut self) {
        let len = self.palette_matches().len();
        if len > 0 {
            self.palette_index = (self.palette_index + 1) % len;
        }
    }

    fn accept_palette(&mut self) {
        if let Some(command) = self.palette_matches().get(self.palette_index) {
            self.input.set(command.name.to_string());
        }
        self.palette_open = false;
    }

    fn history_previous(&mut self) {
        if self.prompt_history.is_empty() {
            return;
        }
        let next = match self.history_index {
            Some(index) => index.saturating_sub(1),
            None => {
                self.history_draft = self.input.text.clone();
                self.prompt_history.len() - 1
            }
        };
        self.history_index = Some(next);
        self.input.set(self.prompt_history[next].clone());
    }

    fn history_next(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < self.prompt_history.len() {
            self.history_index = Some(index + 1);
            self.input.set(self.prompt_history[index + 1].clone());
        } else {
            self.history_index = None;
            self.input.set(self.history_draft.clone());
        }
    }

    fn reset_history_navigation(&mut self) {
        self.history_index = None;
        self.history_draft.clear();
    }

    fn scroll_up(&mut self, amount: u16) {
        self.follow_transcript = false;
        self.transcript_scroll = self.transcript_scroll.saturating_sub(amount);
    }

    fn scroll_down(&mut self, amount: u16) {
        self.transcript_scroll = self
            .transcript_scroll
            .saturating_add(amount)
            .min(self.transcript_max_scroll);
        self.follow_transcript = self.transcript_scroll == self.transcript_max_scroll;
    }

    fn handle_clarification_key(&mut self, key: KeyEvent) -> Option<ClarificationAction> {
        let clarification = self.clarification.as_mut()?;
        if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
            self.clarification = None;
            return Some(ClarificationAction::Cancel);
        }
        if clarification.freeform {
            match key.code {
                KeyCode::Esc => {
                    clarification.freeform = false;
                    clarification.input.clear();
                }
                KeyCode::Left => clarification.input.move_left(),
                KeyCode::Right => clarification.input.move_right(),
                KeyCode::Backspace => clarification.input.backspace(),
                KeyCode::Delete => clarification.input.delete(),
                KeyCode::Enter => {
                    if let Some(response) = clarification.submit() {
                        self.clarification = None;
                        return Some(ClarificationAction::Answer(response));
                    }
                }
                KeyCode::Char(ch)
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::SUPER) =>
                {
                    clarification.input.insert_char(ch);
                }
                _ => {}
            }
            return None;
        }
        match key.code {
            KeyCode::Esc => {
                self.clarification = None;
                Some(ClarificationAction::Cancel)
            }
            KeyCode::Up => {
                clarification.selected = clarification.selected.saturating_sub(1);
                None
            }
            KeyCode::Down => {
                clarification.selected = (clarification.selected + 1)
                    .min(clarification.selection_count().saturating_sub(1));
                None
            }
            KeyCode::Enter => {
                if let Some(response) = clarification.submit() {
                    self.clarification = None;
                    Some(ClarificationAction::Answer(response))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
struct TurnView {
    user: String,
    image_count: usize,
    events: Vec<TurnEvent>,
    stats: Option<String>,
    scratchpad: Option<ScratchpadView>,
    complete: bool,
}

#[derive(Debug)]
struct ScratchpadView {
    summary: String,
    tokens: TokenCount,
}

#[derive(Debug)]
enum TurnEvent {
    Trace {
        label: String,
        text: String,
        state: TraceState,
    },
    Tool(ToolCard),
    Answer(String),
    System(String),
    Error(String),
}

#[derive(Debug, Clone, Copy)]
enum TraceState {
    Live,
    Complete,
    Warning,
}

#[derive(Debug)]
struct ToolCard {
    name: String,
    input: String,
    output: String,
    state: ToolState,
    started: Instant,
    elapsed: Duration,
    expanded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolState {
    Running,
    Success,
    Failure,
}

fn push_trace(turn: &mut TurnView, label: &str, text: &str, state: TraceState) {
    turn.events.push(TurnEvent::Trace {
        label: label.to_string(),
        text: text.to_string(),
        state,
    });
}

fn finish_last_trace(turn: &mut TurnView, label: &str) {
    if let Some(TurnEvent::Trace { state, .. }) =
        turn.events.iter_mut().rev().find(
            |event| matches!(event, TurnEvent::Trace { label: current, .. } if current == label),
        )
    {
        *state = TraceState::Complete;
    }
}

fn finish_tool(turn: &mut TurnView, name: &str, output: String, state: ToolState) {
    if let Some(TurnEvent::Tool(tool)) = turn.events.iter_mut().find(|event| {
        matches!(event, TurnEvent::Tool(tool) if tool.name == name && tool.state == ToolState::Running)
    }) {
        tool.elapsed = tool.started.elapsed();
        tool.output = output;
        tool.state = state;
    } else {
        turn.events.push(TurnEvent::Tool(ToolCard {
            name: name.to_string(),
            input: String::new(),
            output,
            state,
            started: Instant::now(),
            elapsed: Duration::ZERO,
            expanded: false,
        }));
    }
}

fn turn_scratchpad_text(turn_index: usize, scratchpad: &ScratchpadView) -> String {
    format!(
        "interaction {} scratchpad ({} {} tokens from generation response):\n{}",
        turn_index + 1,
        scratchpad.tokens.tokens,
        scratchpad.tokens.source.label(),
        scratchpad.summary.trim()
    )
}

fn inspector_text(key: InspectorKey, snapshot: &ReplSnapshot) -> String {
    match key {
        InspectorKey::Help => help_text(),
        InspectorKey::Stats => snapshot.stats.clone(),
        InspectorKey::Context => snapshot.context.clone(),
        InspectorKey::Scratchpad => snapshot.scratchpad.clone(),
        InspectorKey::Manifest => snapshot.manifest.clone(),
        InspectorKey::Skills => snapshot.skills.clone(),
        InspectorKey::Debug => snapshot.debug.clone(),
    }
}

fn help_text() -> String {
    let mut lines = vec!["commands".to_string()];
    lines.extend(
        COMMANDS
            .iter()
            .map(|command| format!("  {:<12} {}", command.name, command.description)),
    );
    lines.extend([
        String::new(),
        "controls".to_string(),
        "  Ctrl+K / Ctrl+O / F1  command palette".to_string(),
        "  Left click run command · mouse wheel select".to_string(),
        "  Click model/header/inspect controls and model rows".to_string(),
        "  Click ◇ scratchpad or ▸ tool rows for interaction details".to_string(),
        "  Alt+T trace · Alt+E inspect · Alt+M models · Alt+C clear".to_string(),
        "  PageUp/PageDown scroll · End follow latest".to_string(),
        "  Enter send · Shift+Enter newline · Esc exit/cancel".to_string(),
    ]);
    lines.join("\n")
}

fn summarize_tool_input(input: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(input) else {
        return truncate_line(input, 100);
    };
    if let Some(commands) = value.get("commands").and_then(serde_json::Value::as_array) {
        let commands = commands
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect::<Vec<_>>()
            .join(" && ");
        if !commands.is_empty() {
            return truncate_line(&commands, 100);
        }
    }
    for key in ["path", "query", "name", "prompt"] {
        if let Some(value) = value.get(key).and_then(serde_json::Value::as_str) {
            return truncate_line(value, 100);
        }
    }
    truncate_line(input, 100)
}

fn truncate_line(text: &str, max: usize) -> String {
    let text = text.replace(['\r', '\n'], " ");
    if text.chars().count() <= max {
        return text;
    }
    format!(
        "{}…",
        text.chars().take(max.saturating_sub(1)).collect::<String>()
    )
}

struct TerminalSession {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalSession {
    fn enter() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        let mut output = stdout();
        if let Err(error) = execute!(
            output,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture,
            Hide
        ) {
            let _ = disable_raw_mode();
            return Err(error.into());
        }
        let backend = CrosstermBackend::new(output);
        let terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(error) => {
                let mut output = stdout();
                let _ = execute!(
                    output,
                    Show,
                    DisableMouseCapture,
                    DisableBracketedPaste,
                    LeaveAlternateScreen
                );
                let _ = disable_raw_mode();
                return Err(error.into());
            }
        };
        Ok(Self { terminal })
    }

    fn draw(&mut self, state: &mut UiState) -> std::io::Result<()> {
        self.terminal.draw(|frame| render(frame, state))?;
        Ok(())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            Show,
            DisableMouseCapture,
            DisableBracketedPaste,
            LeaveAlternateScreen
        );
        let _ = self.terminal.show_cursor();
    }
}

fn render(frame: &mut ratatui::Frame<'_>, state: &mut UiState) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(VY_BG)), area);

    if area.width < 64 || area.height < 18 {
        let warning = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                "VYRN",
                Style::default().fg(VY_VIOLET).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "terminal too small · resize to at least 64 × 18",
                Style::default().fg(VY_AMBER),
            )),
        ]))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(VY_BORDER_STRONG)),
        );
        frame.render_widget(warning, centered(area, 58, 7));
        return;
    }

    let chunks = interface_chunks(area, state);

    render_header(frame, chunks[0], state);
    render_context(frame, chunks[1], state);
    if chunks[2].height > 0 {
        render_inspect_strip(frame, chunks[2], state);
    }
    if chunks[3].height > 0 {
        render_inspector(frame, chunks[3], state);
    }
    render_transcript(frame, chunks[4], state);
    render_composer(frame, chunks[5], state);
    render_footer(frame, chunks[6], state);

    if state.palette_open && !state.running {
        render_palette(frame, chunks[5], state);
    }
    if state.model_picker.is_some() {
        render_model_picker(frame, area, state);
    }
    if state.clarification.is_some() {
        render_clarification(frame, area, state);
    }
}

fn interface_chunks(area: Rect, state: &UiState) -> std::rc::Rc<[Rect]> {
    let inspect_height = if state.inspect_visible {
        inspect_strip_height(area.width)
    } else {
        0
    };
    let fixed_height = 3_u16
        .saturating_add(3)
        .saturating_add(inspect_height)
        .saturating_add(3)
        .saturating_add(4)
        .saturating_add(1);
    let available_inspector_height = area.height.saturating_sub(fixed_height).min(8);
    let inspector_height =
        if state.inspect_visible && state.inspector.is_some() && available_inspector_height >= 3 {
            available_inspector_height
        } else {
            0
        };
    Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(inspect_height),
            Constraint::Length(inspector_height),
            Constraint::Min(3),
            Constraint::Length(4),
            Constraint::Length(1),
        ])
        .split(area)
}

fn append_click_text(
    regions: &mut Vec<ClickRegion>,
    cursor: &mut u16,
    area: Rect,
    text: &str,
    target: Option<UiClickTarget>,
) {
    let start = *cursor;
    let width = u16::try_from(Span::raw(text).width()).unwrap_or(u16::MAX);
    *cursor = cursor.saturating_add(width);
    let Some(target) = target else {
        return;
    };
    let left = start.max(area.x);
    let right = (*cursor).min(area.right());
    if area.height > 0 && left < right {
        regions.push(ClickRegion {
            area: Rect::new(left, area.y, right - left, 1),
            target,
        });
    }
}

#[derive(Debug)]
struct HeaderPresentation {
    cwd: String,
    model: String,
    debug_path: Option<String>,
    shortcuts: bool,
}

fn display_width(text: &str) -> u16 {
    u16::try_from(Span::raw(text).width()).unwrap_or(u16::MAX)
}

fn compact_display(text: &str, max_width: u16) -> String {
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 0 {
        return String::new();
    }
    if max_width == 1 {
        return "…".to_string();
    }

    let mut suffix = String::new();
    for character in text.chars().rev() {
        let candidate = format!("{character}{suffix}");
        if display_width(&candidate).saturating_add(1) > max_width {
            break;
        }
        suffix = candidate;
    }
    format!("…{suffix}")
}

fn header_presentation(area: Rect, state: &UiState) -> HeaderPresentation {
    // Everything except cwd/model metadata, debug metadata, and shortcut hints.
    const FIXED_CONTROL_WIDTH: u16 = 52;
    const MIN_CWD_WIDTH: u16 = 2;
    const MAX_MODEL_WIDTH: u16 = 20;
    const MAX_CWD_WIDTH: u16 = 30;
    const DEBUG_PREFIX_WIDTH: u16 = 12;
    const MAX_DEBUG_PATH_WIDTH: u16 = 24;
    const SHORTCUTS_WIDTH: u16 = 15;

    let metadata_width = area.width.saturating_sub(FIXED_CONTROL_WIDTH);
    let model_width = display_width(&state.snapshot.model_name)
        .min(MAX_MODEL_WIDTH)
        .min(metadata_width.saturating_sub(MIN_CWD_WIDTH));
    let model = compact_display(&state.snapshot.model_name, model_width);
    let cwd_width = metadata_width
        .saturating_sub(display_width(&model))
        .min(MAX_CWD_WIDTH);
    let cwd = compact_display(&state.snapshot.cwd, cwd_width);
    let mut used = FIXED_CONTROL_WIDTH
        .saturating_add(display_width(&cwd))
        .saturating_add(display_width(&model));
    let shortcuts =
        state.show_shortcuts_until.is_some() && used.saturating_add(SHORTCUTS_WIDTH) <= area.width;
    if shortcuts {
        used = used.saturating_add(SHORTCUTS_WIDTH);
    }
    let debug_path = state.snapshot.debug_path.as_deref().and_then(|path| {
        let available = area.width.saturating_sub(used);
        (available > DEBUG_PREFIX_WIDTH).then(|| {
            compact_display(
                path,
                available
                    .saturating_sub(DEBUG_PREFIX_WIDTH)
                    .min(MAX_DEBUG_PATH_WIDTH),
            )
        })
    });

    HeaderPresentation {
        cwd,
        model,
        debug_path,
        shortcuts,
    }
}

fn header_click_regions(area: Rect, state: &UiState) -> Vec<ClickRegion> {
    let mut regions = Vec::new();
    let mut cursor = area.x;
    let display = header_presentation(area, state);
    append_click_text(&mut regions, &mut cursor, area, " VYRN ", None);
    append_click_text(&mut regions, &mut cursor, area, "  │  ", None);
    append_click_text(&mut regions, &mut cursor, area, "●", None);
    append_click_text(&mut regions, &mut cursor, area, "  cwd ", None);
    append_click_text(&mut regions, &mut cursor, area, &display.cwd, None);
    append_click_text(&mut regions, &mut cursor, area, "  │  ", None);
    append_click_text(
        &mut regions,
        &mut cursor,
        area,
        &display.model,
        Some(UiClickTarget::OpenModels),
    );
    if let Some(path) = &display.debug_path {
        append_click_text(&mut regions, &mut cursor, area, "  │  ", None);
        append_click_text(&mut regions, &mut cursor, area, " debug ", None);
        append_click_text(&mut regions, &mut cursor, area, path, None);
    }
    append_click_text(&mut regions, &mut cursor, area, "  │  ", None);
    append_click_text(
        &mut regions,
        &mut cursor,
        area,
        if state.trace_visible {
            "trace on"
        } else {
            "trace off"
        },
        Some(UiClickTarget::ToggleTrace),
    );
    if display.shortcuts {
        append_click_text(&mut regions, &mut cursor, area, " [⌥T]", None);
    }
    append_click_text(
        &mut regions,
        &mut cursor,
        area,
        "  inspect",
        Some(UiClickTarget::ToggleInspect),
    );
    if display.shortcuts {
        append_click_text(&mut regions, &mut cursor, area, " [⌥E]", None);
    }
    append_click_text(
        &mut regions,
        &mut cursor,
        area,
        "  clear",
        Some(UiClickTarget::Clear),
    );
    regions
}

fn inspect_button_specs() -> [(UiClickTarget, &'static str); 7] {
    [
        (UiClickTarget::Inspector(InspectorKey::Help), "help"),
        (UiClickTarget::Inspector(InspectorKey::Stats), "stats"),
        (UiClickTarget::Inspector(InspectorKey::Context), "context"),
        (
            UiClickTarget::Inspector(InspectorKey::Scratchpad),
            "scratchpad",
        ),
        (UiClickTarget::Inspector(InspectorKey::Manifest), "manifest"),
        (UiClickTarget::Inspector(InspectorKey::Skills), "skills"),
        (UiClickTarget::Refresh, "refresh"),
    ]
}

fn inspect_button_width(label: &str, horizontal_padding: u16) -> u16 {
    u16::try_from(Span::raw(label).width())
        .unwrap_or(u16::MAX)
        .saturating_add(2)
        .saturating_add(horizontal_padding.saturating_mul(2))
}

fn inspect_button_spacing(width: u16) -> (u16, u16) {
    let specs = inspect_button_specs();
    let labels_and_borders = specs
        .iter()
        .map(|(_, label)| inspect_button_width(label, 0))
        .fold(0_u16, u16::saturating_add);
    let button_gaps = specs.len().saturating_sub(1) as u16;
    let padded = labels_and_borders.saturating_add(specs.len() as u16 * 2);
    if padded.saturating_add(button_gaps) <= width {
        (1, 1)
    } else if padded <= width {
        (1, 0)
    } else if labels_and_borders.saturating_add(button_gaps) <= width {
        (0, 1)
    } else {
        (0, 0)
    }
}

fn inspect_button_layout(width: u16) -> Vec<(UiClickTarget, &'static str, Rect)> {
    if width == 0 {
        return Vec::new();
    }
    let (horizontal_padding, gap) = inspect_button_spacing(width);
    let mut layout = Vec::new();
    let mut x = 0_u16;
    let mut y = 0_u16;
    for (target, label) in inspect_button_specs() {
        let button_width = inspect_button_width(label, horizontal_padding).min(width);
        if x > 0 && x.saturating_add(gap).saturating_add(button_width) > width {
            x = 0;
            y = y.saturating_add(3);
        } else if x > 0 {
            x = x.saturating_add(gap);
        }
        layout.push((target, label, Rect::new(x, y, button_width, 3)));
        x = x.saturating_add(button_width);
    }
    layout
}

fn inspect_strip_height(width: u16) -> u16 {
    inspect_button_layout(width)
        .last()
        .map_or(0, |(_, _, area)| area.bottom())
}

fn inspect_strip_click_regions(area: Rect) -> Vec<ClickRegion> {
    if area.width == 0 || area.height < 3 {
        return Vec::new();
    }
    inspect_button_layout(area.width)
        .into_iter()
        .filter_map(|(target, _, button)| {
            let button = Rect::new(
                area.x.saturating_add(button.x),
                area.y.saturating_add(button.y),
                button.width,
                button.height,
            );
            (button.bottom() <= area.bottom()).then_some(ClickRegion {
                area: button,
                target,
            })
        })
        .collect()
}

fn chrome_click_target(
    chunks: &[Rect],
    state: &UiState,
    column: u16,
    row: u16,
) -> Option<UiClickTarget> {
    header_click_regions(chunks[0], state)
        .into_iter()
        .chain(inspect_strip_click_regions(chunks[2]))
        .find(|region| region.contains(column, row))
        .map(|region| region.target)
}

fn transcript_click_target(state: &UiState, column: u16, row: u16) -> Option<UiClickTarget> {
    state
        .transcript_click_regions
        .iter()
        .find(|region| region.contains(column, row))
        .map(|region| region.target)
}

fn render_header(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let status_color = if state.running { VY_VIOLET } else { VY_SUCCESS };
    let display = header_presentation(area, state);
    let mut spans = vec![
        Span::styled(
            " VYRN ",
            Style::default()
                .fg(VY_VIOLET_HOVER)
                .bg(VY_SURFACE_RAISED)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  │  ", Style::default().fg(VY_BORDER)),
        Span::styled("●", Style::default().fg(status_color)),
        Span::styled("  cwd ", Style::default().fg(VY_MUTED)),
        Span::styled(display.cwd, Style::default().fg(VY_TEXT)),
        Span::styled("  │  ", Style::default().fg(VY_BORDER)),
        Span::styled(display.model, Style::default().fg(VY_TECH_STRONG)),
    ];
    if let Some(path) = display.debug_path {
        spans.extend([
            Span::styled("  │  ", Style::default().fg(VY_BORDER)),
            Span::styled(" debug ", Style::default().fg(VY_AMBER)),
            Span::styled(path, Style::default().fg(VY_DIM)),
        ]);
    }
    let shortcuts = display.shortcuts;
    spans.extend([
        Span::styled("  │  ", Style::default().fg(VY_BORDER)),
        Span::styled(
            if state.trace_visible {
                "trace on"
            } else {
                "trace off"
            },
            Style::default().fg(if state.trace_visible { VY_TECH } else { VY_DIM }),
        ),
        Span::styled(
            if shortcuts { " [⌥T]" } else { "" },
            Style::default().fg(VY_MUTED),
        ),
        Span::styled(
            "  inspect",
            header_action_style(
                state,
                UiClickTarget::ToggleInspect,
                if state.inspect_visible {
                    VY_TECH_STRONG
                } else {
                    VY_MUTED
                },
            ),
        ),
        Span::styled(
            if shortcuts { " [⌥E]" } else { "" },
            Style::default().fg(VY_MUTED),
        ),
        Span::styled(
            "  clear",
            header_action_style(state, UiClickTarget::Clear, VY_MUTED),
        ),
        Span::styled(
            if shortcuts { " [⌥C]" } else { "" },
            Style::default().fg(VY_MUTED),
        ),
    ]);
    let header = Paragraph::new(Line::from(spans))
        .block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(VY_BORDER)),
        )
        .style(Style::default().bg(VY_SURFACE));
    frame.render_widget(header, area);
}

fn click_feedback_active(state: &UiState, target: UiClickTarget) -> bool {
    state
        .click_feedback
        .is_some_and(|(clicked, _)| clicked == target)
}

fn header_action_style(state: &UiState, target: UiClickTarget, color: Color) -> Style {
    if click_feedback_active(state, target) {
        Style::default()
            .fg(VY_VIOLET_HOVER)
            .bg(VY_SURFACE_RAISED)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(color)
    }
}

fn render_context(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let used = state.snapshot.context_used;
    let limit = state.snapshot.context_limit.max(1);
    let available = limit.saturating_sub(used);
    let meta = Line::from(vec![
        Span::styled("context ", Style::default().fg(VY_MUTED)),
        Span::styled(format_number(used as isize), Style::default().fg(VY_TEXT)),
        Span::styled(
            format!(" / {}", format_number(limit as isize)),
            Style::default().fg(VY_MUTED),
        ),
        Span::styled("  │  system ", Style::default().fg(VY_DIM)),
        Span::styled(
            format_number(state.snapshot.context_system as isize),
            Style::default().fg(VY_TECH_STRONG),
        ),
        Span::styled("  history ", Style::default().fg(VY_DIM)),
        Span::styled(
            format_number(state.snapshot.context_history as isize),
            Style::default().fg(VY_TECH),
        ),
        Span::styled("  scratch ", Style::default().fg(VY_DIM)),
        Span::styled(
            format_number(state.snapshot.context_scratch as isize),
            Style::default().fg(VY_VIOLET_HOVER),
        ),
        Span::styled("  │  available ", Style::default().fg(VY_DIM)),
        Span::styled(
            format_number(available as isize),
            Style::default().fg(if used > limit.saturating_mul(85) / 100 {
                VY_AMBER
            } else {
                VY_SUCCESS
            }),
        ),
    ]);
    let outer = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(VY_BORDER))
        .style(Style::default().bg(VY_SURFACE));
    let inner = outer.inner(area).inner(Margin {
        horizontal: 1,
        vertical: 0,
    });
    frame.render_widget(outer, area);
    if inner.height < 2 {
        return;
    }
    frame.render_widget(
        Paragraph::new(meta),
        Rect::new(inner.x, inner.y, inner.width, 1),
    );
    let bar = Rect::new(inner.x, inner.y + 1, inner.width, 1);
    frame.render_widget(
        Block::default().style(Style::default().bg(VY_SURFACE_RAISED)),
        bar,
    );
    let system_width = ratio_width(bar.width, state.snapshot.context_system, limit);
    let history_width = ratio_width(bar.width, state.snapshot.context_history, limit)
        .min(bar.width.saturating_sub(system_width));
    let scratch_width = ratio_width(bar.width, state.snapshot.context_scratch, limit)
        .min(bar.width.saturating_sub(system_width + history_width));
    if system_width > 0 {
        frame.render_widget(
            Block::default().style(Style::default().bg(VY_DIM)),
            Rect::new(bar.x, bar.y, system_width, 1),
        );
    }
    if history_width > 0 {
        frame.render_widget(
            Block::default().style(Style::default().bg(VY_TECH)),
            Rect::new(bar.x + system_width, bar.y, history_width, 1),
        );
    }
    if scratch_width > 0 {
        frame.render_widget(
            Block::default().style(Style::default().bg(VY_VIOLET)),
            Rect::new(
                bar.x + system_width + history_width,
                bar.y,
                scratch_width,
                1,
            ),
        );
    }
}

fn render_inspect_strip(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    frame.render_widget(
        Block::default().style(Style::default().bg(VY_SURFACE)),
        area,
    );
    for (target, label, button) in inspect_button_layout(area.width) {
        let button = Rect::new(
            area.x.saturating_add(button.x),
            area.y.saturating_add(button.y),
            button.width,
            button.height,
        );
        if button.bottom() > area.bottom() {
            continue;
        }
        let selected =
            matches!(target, UiClickTarget::Inspector(key) if state.inspector == Some(key));
        let clicked = click_feedback_active(state, target);
        let highlighted = selected || clicked;
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if highlighted {
                VY_VIOLET_HOVER
            } else {
                VY_BORDER_STRONG
            }))
            .style(Style::default().bg(if highlighted {
                VY_SURFACE_RAISED
            } else {
                VY_SURFACE
            }));
        let inner = block.inner(button);
        frame.render_widget(block, button);
        frame.render_widget(
            Paragraph::new(label).alignment(Alignment::Center).style(
                Style::default()
                    .fg(if highlighted { VY_TEXT } else { VY_MUTED })
                    .add_modifier(if selected {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
            inner,
        );
    }
}

fn render_inspector(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let Some(key) = state.inspector else {
        return;
    };
    let label = state
        .inspected_turn
        .filter(|_| key == InspectorKey::Scratchpad)
        .map(|index| format!("scratchpad · interaction {}", index + 1))
        .unwrap_or_else(|| key.label().to_string());
    let title = Line::from(vec![
        Span::styled(" inspect · ", Style::default().fg(VY_DIM)),
        Span::styled(label, Style::default().fg(VY_TEXT)),
        Span::styled("  Esc close ", Style::default().fg(VY_DIM)),
    ]);
    let panel = Paragraph::new(state.inspector_body.clone())
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(VY_TECH_STRONG).bg(VY_SURFACE))
        .block(
            Block::default()
                .title(title)
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(VY_BORDER)),
        );
    frame.render_widget(panel, area);
}

fn render_transcript(frame: &mut ratatui::Frame<'_>, area: Rect, state: &mut UiState) {
    let rows = transcript_rows(state);
    let heights = rows
        .iter()
        .map(|row| wrapped_line_height(&row.line, area.width))
        .collect::<Vec<_>>();
    let line_count = heights
        .iter()
        .copied()
        .sum::<usize>()
        .min(u16::MAX as usize) as u16;
    state.transcript_max_scroll = line_count.saturating_sub(area.height);
    if state.follow_transcript {
        state.transcript_scroll = state.transcript_max_scroll;
    } else {
        state.transcript_scroll = state.transcript_scroll.min(state.transcript_max_scroll);
    }
    frame.render_widget(Block::default().style(Style::default().bg(VY_BG)), area);
    state.transcript_click_regions.clear();

    let viewport_start = usize::from(state.transcript_scroll);
    let mut logical_y = 0_usize;
    let mut screen_y = area.y;
    for (transcript_row, height) in rows.into_iter().zip(heights) {
        let logical_end = logical_y.saturating_add(height);
        if logical_end <= viewport_start {
            logical_y = logical_end;
            continue;
        }
        if screen_y >= area.bottom() {
            break;
        }

        let skipped = viewport_start.saturating_sub(logical_y);
        let visible_height = height
            .saturating_sub(skipped)
            .min(usize::from(area.bottom().saturating_sub(screen_y)));
        let row = Rect::new(
            area.x,
            screen_y,
            area.width,
            u16::try_from(visible_height).unwrap_or(u16::MAX),
        );
        let background = transcript_row.line.style.bg.unwrap_or(VY_BG);
        frame.render_widget(
            Paragraph::new(transcript_row.line)
                .wrap(Wrap { trim: false })
                .scroll((u16::try_from(skipped).unwrap_or(u16::MAX), 0))
                .style(Style::default().fg(VY_TEXT).bg(background)),
            row,
        );
        if let Some(target) = transcript_row.target {
            state
                .transcript_click_regions
                .push(ClickRegion { area: row, target });
        }
        screen_y = screen_y.saturating_add(row.height);
        logical_y = logical_end;
    }
}

fn wrapped_line_height(line: &Line<'_>, width: u16) -> usize {
    line.width().max(1).div_ceil(usize::from(width.max(1)))
}

struct TranscriptRow {
    line: Line<'static>,
    target: Option<UiClickTarget>,
}

impl TranscriptRow {
    fn new(line: impl Into<Line<'static>>) -> Self {
        Self {
            line: line.into(),
            target: None,
        }
    }

    fn clickable(line: impl Into<Line<'static>>, target: UiClickTarget) -> Self {
        Self {
            line: line.into(),
            target: Some(target),
        }
    }
}

fn transcript_rows(state: &UiState) -> Vec<TranscriptRow> {
    if state.turns.is_empty() {
        return vec![
            TranscriptRow::new(""),
            TranscriptRow::new(Span::styled(
                "  vyrn full-screen preview",
                Style::default()
                    .fg(VY_VIOLET_HOVER)
                    .add_modifier(Modifier::BOLD),
            )),
            TranscriptRow::new(Span::styled(
                "  Send a request below.",
                Style::default().fg(VY_MUTED),
            )),
        ];
    }
    let mut rows = vec![TranscriptRow::new("")];
    for (turn_index, turn) in state.turns.iter().enumerate() {
        if turn_index > 0 {
            rows.push(TranscriptRow::new(""));
            rows.push(TranscriptRow::new(""));
        }
        let image_suffix = if turn.image_count > 0 {
            format!("  [{} image(s)]", turn.image_count)
        } else {
            String::new()
        };
        for (index, user_line) in turn.user.lines().enumerate() {
            rows.push(TranscriptRow::new(
                Line::from(vec![
                    Span::styled(
                        if index == 0 { " › " } else { "   " },
                        Style::default().fg(VY_TECH),
                    ),
                    Span::styled(
                        format!("{user_line}{}", if index == 0 { &image_suffix } else { "" }),
                        Style::default().fg(VY_TEXT),
                    ),
                ])
                .style(Style::default().bg(VY_SURFACE_RAISED)),
            ));
        }
        rows.push(TranscriptRow::new(""));
        for (event_index, event) in turn.events.iter().enumerate() {
            match event {
                TurnEvent::Trace {
                    label,
                    text,
                    state: trace_state,
                } if state.trace_visible => {
                    let color = match trace_state {
                        TraceState::Live => VY_VIOLET,
                        TraceState::Complete => VY_SUCCESS,
                        TraceState::Warning => VY_AMBER,
                    };
                    rows.push(TranscriptRow::new(Line::from(vec![
                        Span::styled("     │ ", Style::default().fg(VY_BORDER_STRONG)),
                        Span::styled("● ", Style::default().fg(color)),
                        Span::styled(format!("{label:<7}"), Style::default().fg(VY_DIM)),
                        Span::styled(text.clone(), Style::default().fg(VY_MUTED)),
                    ])));
                }
                TurnEvent::Tool(tool) if state.trace_visible => append_tool_rows(
                    &mut rows,
                    tool,
                    turn_index,
                    event_index,
                    state.inspect_visible,
                ),
                TurnEvent::Answer(answer) => append_answer_rows(&mut rows, answer),
                TurnEvent::System(message) => {
                    for line in message.lines() {
                        rows.push(TranscriptRow::new(Line::from(vec![
                            Span::styled("     · ", Style::default().fg(VY_VIOLET)),
                            Span::styled(line.to_string(), Style::default().fg(VY_MUTED)),
                        ])));
                    }
                }
                TurnEvent::Error(message) => {
                    for line in message.lines() {
                        rows.push(TranscriptRow::new(Line::from(vec![
                            Span::styled("     ! ", Style::default().fg(VY_RED)),
                            Span::styled(line.to_string(), Style::default().fg(VY_RED)),
                        ])));
                    }
                }
                _ => {}
            }
        }
        if !turn.complete
            && let Some(activity) = &state.activity
        {
            let pulse = ["·  ", "·· ", "···", " ··", "  ·", "   "][state.spinner_frame % 6];
            rows.push(TranscriptRow::new(Line::from(vec![
                Span::styled("     ", Style::default()),
                Span::styled(format!("{pulse}  "), Style::default().fg(VY_VIOLET)),
                Span::styled(activity.clone(), Style::default().fg(VY_MUTED)),
            ])));
        }
        if state.inspect_visible {
            let scratchpad_meta = turn.scratchpad.as_ref().map_or_else(
                || "none  [click inspect]".to_string(),
                |scratchpad| {
                    format!(
                        "{} {} tokens  [click inspect]",
                        scratchpad.tokens.tokens,
                        scratchpad.tokens.source.label()
                    )
                },
            );
            rows.push(TranscriptRow::clickable(
                Line::from(vec![
                    Span::styled("     ◇ ", Style::default().fg(VY_VIOLET_HOVER)),
                    Span::styled("scratchpad", Style::default().fg(VY_TECH_STRONG)),
                    Span::styled(format!(" · {scratchpad_meta}"), Style::default().fg(VY_DIM)),
                ]),
                UiClickTarget::TurnScratchpad(turn_index),
            ));
        }
        if let Some(stats) = &turn.stats {
            rows.push(TranscriptRow::new(Line::from(vec![
                Span::styled("     ✓ ", Style::default().fg(VY_SUCCESS)),
                Span::styled(stats.clone(), Style::default().fg(VY_DIM)),
            ])));
        }
    }
    rows
}

#[cfg(test)]
fn transcript_lines(state: &UiState) -> Vec<Line<'static>> {
    transcript_rows(state)
        .into_iter()
        .map(|row| row.line)
        .collect()
}

fn append_answer_rows(rows: &mut Vec<TranscriptRow>, answer: &str) {
    let mut in_code = false;
    for raw in answer.lines() {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        let rendered = trimmed
            .strip_prefix("### ")
            .or_else(|| trimmed.strip_prefix("## "))
            .or_else(|| trimmed.strip_prefix("# "))
            .unwrap_or(raw);
        rows.push(TranscriptRow::new(
            Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(
                    rendered.to_string(),
                    Style::default()
                        .fg(if in_code { VY_TECH_STRONG } else { VY_TEXT })
                        .bg(if in_code {
                            VY_SURFACE_RAISED
                        } else {
                            VY_SURFACE
                        })
                        .add_modifier(if raw != rendered {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
            ])
            .style(Style::default().bg(VY_SURFACE)),
        ));
    }
}

fn append_tool_rows(
    rows: &mut Vec<TranscriptRow>,
    tool: &ToolCard,
    turn_index: usize,
    event_index: usize,
    inspect_visible: bool,
) {
    let (state_label, state_color) = match tool.state {
        ToolState::Running => ("running", VY_VIOLET),
        ToolState::Success => ("ok", VY_SUCCESS),
        ToolState::Failure => ("error", VY_RED),
    };
    let elapsed = if tool.state == ToolState::Running {
        tool.started.elapsed()
    } else {
        tool.elapsed
    };
    let target = UiClickTarget::ToolDetails {
        turn_index,
        event_index,
    };
    let summary = summarize_tool_input(&tool.input);
    let header = Line::from(vec![
        Span::styled(
            if inspect_visible {
                if tool.expanded {
                    "     ▾  "
                } else {
                    "     ▸  "
                }
            } else {
                "     ·  "
            },
            Style::default().fg(if tool.expanded {
                VY_VIOLET_HOVER
            } else {
                VY_BORDER_STRONG
            }),
        ),
        Span::styled(
            format!("tool.{}", tool.name),
            Style::default().fg(VY_VIOLET_HOVER),
        ),
        Span::styled("  ", Style::default()),
        Span::styled(summary, Style::default().fg(VY_TECH_STRONG)),
        Span::styled(
            format!("  {:.2}s  ", elapsed.as_secs_f64()),
            Style::default().fg(VY_DIM),
        ),
        Span::styled(state_label, Style::default().fg(state_color)),
        Span::styled(
            if inspect_visible { "  [click]" } else { "" },
            Style::default().fg(VY_DIM),
        ),
    ]);
    if inspect_visible {
        rows.push(TranscriptRow::clickable(header, target));
    } else {
        rows.push(TranscriptRow::new(header));
    }
    if !inspect_visible || !tool.expanded {
        return;
    }
    if !tool.input.trim().is_empty() {
        rows.push(TranscriptRow::new(Line::from(vec![
            Span::styled("        input  ", Style::default().fg(VY_DIM)),
            Span::styled(tool.input.clone(), Style::default().fg(VY_TECH_STRONG)),
        ])));
    }
    if !tool.output.trim().is_empty() {
        for output in tool.output.lines().take(8) {
            rows.push(TranscriptRow::new(Line::from(vec![
                Span::styled("        output ", Style::default().fg(VY_DIM)),
                Span::styled(output.to_string(), Style::default().fg(VY_TECH)),
            ])));
        }
        if tool.output.lines().count() > 8 {
            rows.push(TranscriptRow::new(Line::from(vec![
                Span::styled("               ", Style::default()),
                Span::styled("… preview truncated", Style::default().fg(VY_DIM)),
            ])));
        }
    }
    rows.push(TranscriptRow::new(Span::styled(
        "        └─────────────────────────────────────",
        Style::default().fg(VY_BORDER_STRONG),
    )));
}

fn render_composer(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(VY_BORDER_STRONG))
        .style(Style::default().bg(VY_SURFACE));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let placeholder = if state.running {
        "steer the active turn…"
    } else {
        "message vyrn…  / for commands"
    };
    let text = if state.input.text.is_empty() {
        Line::from(vec![
            Span::styled(" › ", Style::default().fg(VY_TECH)),
            Span::styled(placeholder, Style::default().fg(VY_DIM)),
        ])
    } else {
        Line::from(vec![
            Span::styled(" › ", Style::default().fg(VY_TECH)),
            Span::styled(state.input.text.clone(), Style::default().fg(VY_TEXT)),
        ])
    };
    frame.render_widget(
        Paragraph::new(text).wrap(Wrap { trim: false }),
        Rect::new(
            inner.x,
            inner.y,
            inner.width,
            inner.height.saturating_sub(1),
        ),
    );
    let hint = if state.running {
        "enter steer  ·  shift+enter newline  ·  esc cancel  ·  page↑/↓ scroll"
    } else {
        "enter send  ·  shift+enter newline  ·  ctrl+k commands  ·  ctrl+v attach/paste  ·  page↑/↓ scroll"
    };
    let image_hint = if state.images.is_empty() {
        String::new()
    } else {
        format!("  ·  {} image(s) attached", state.images.len())
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("   ", Style::default()),
            Span::styled(hint, Style::default().fg(VY_DIM)),
            Span::styled(image_hint, Style::default().fg(VY_TECH)),
        ])),
        Rect::new(
            inner.x,
            inner.y + inner.height.saturating_sub(1),
            inner.width,
            1,
        ),
    );

    if state.model_picker.is_none() && state.clarification.is_none() {
        let cursor_prefix = state
            .input
            .text
            .chars()
            .take(state.input.cursor)
            .collect::<String>();
        let last_line = cursor_prefix.rsplit('\n').next().unwrap_or_default();
        let row = cursor_prefix.matches('\n').count() as u16;
        let x = inner.x
            + 3
            + last_line
                .chars()
                .count()
                .min(inner.width.saturating_sub(4) as usize) as u16;
        let y = inner.y + row.min(inner.height.saturating_sub(2));
        frame.set_cursor_position((x, y));
    }
}

fn render_footer(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let last = if state.last_latency.is_zero() {
        "—".to_string()
    } else {
        format!("{:.2}s", state.last_latency.as_secs_f64())
    };
    let line = Line::from(vec![
        Span::styled(" turns ", Style::default().fg(VY_DIM)),
        Span::styled(
            (state.snapshot.turns + usize::from(state.running)).to_string(),
            Style::default().fg(VY_MUTED),
        ),
        Span::styled("  spent ", Style::default().fg(VY_DIM)),
        Span::styled(
            format!("{}t", format_number(state.snapshot.session_spent as isize)),
            Style::default().fg(VY_MUTED),
        ),
        Span::styled("  history saved ", Style::default().fg(VY_DIM)),
        Span::styled(
            format_number(state.snapshot.turn_saved),
            Style::default().fg(VY_SUCCESS),
        ),
        Span::styled("  │  session saved ", Style::default().fg(VY_DIM)),
        Span::styled(
            format_number(state.snapshot.session_saved),
            Style::default().fg(VY_SUCCESS),
        ),
        Span::styled("  tools ", Style::default().fg(VY_DIM)),
        Span::styled(state.tool_count.to_string(), Style::default().fg(VY_TECH)),
        Span::styled("  last ", Style::default().fg(VY_DIM)),
        Span::styled(last, Style::default().fg(VY_MUTED)),
        Span::styled("  │  ", Style::default().fg(VY_BORDER)),
        Span::styled(state.status.clone(), Style::default().fg(VY_MUTED)),
    ]);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().bg(VY_SURFACE_RAISED)),
        area,
    );
}

fn render_palette(frame: &mut ratatui::Frame<'_>, composer: Rect, state: &UiState) {
    let matches = state.palette_matches();
    if matches.is_empty() {
        return;
    }
    let area = command_palette_area(composer, matches.len());
    let lines = matches
        .iter()
        .enumerate()
        .map(|(index, command)| {
            let selected = index == state.palette_index;
            Line::from(vec![
                Span::styled(
                    format!(" {:<13}", command.name),
                    Style::default()
                        .fg(if selected { VY_VIOLET_HOVER } else { VY_VIOLET })
                        .bg(if selected {
                            VY_SURFACE_RAISED
                        } else {
                            VY_SURFACE
                        }),
                ),
                Span::styled(
                    command.description,
                    Style::default()
                        .fg(if selected { VY_TEXT } else { VY_MUTED })
                        .bg(if selected {
                            VY_SURFACE_RAISED
                        } else {
                            VY_SURFACE
                        }),
                ),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(Clear, area);
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(" commands · click/Enter run · ↑↓ select · Tab accept ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(VY_BORDER_STRONG))
                .style(Style::default().bg(VY_SURFACE)),
        ),
        area,
    );
}

fn command_palette_area(composer: Rect, command_count: usize) -> Rect {
    let height = u16::try_from(command_count)
        .unwrap_or(u16::MAX)
        .saturating_add(2)
        .min(14);
    Rect::new(
        composer.x,
        composer.y.saturating_sub(height),
        composer.width,
        height,
    )
}

fn command_palette_index_at(
    area: Rect,
    command_count: usize,
    column: u16,
    row: u16,
) -> Option<usize> {
    let inner_left = area.x.saturating_add(1);
    let inner_right = area.right().saturating_sub(1);
    let inner_top = area.y.saturating_add(1);
    let inner_bottom = area.bottom().saturating_sub(1);
    if column < inner_left || column >= inner_right || row < inner_top || row >= inner_bottom {
        return None;
    }
    let index = usize::from(row - inner_top);
    (index < command_count).then_some(index)
}

fn render_model_picker(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let Some(picker) = &state.model_picker else {
        return;
    };
    let layout = model_picker_layout(area, picker);
    let popup = layout.area;
    let mut lines = Vec::new();
    if picker.models.is_empty() {
        lines.push(Line::from(Span::styled(
            "No configured model profiles.",
            Style::default().fg(VY_AMBER),
        )));
    } else {
        let end = layout
            .start
            .saturating_add(layout.len)
            .min(picker.models.len());
        for (index, model) in picker.models[layout.start..end].iter().enumerate() {
            let index = layout.start + index;
            let selected = index == picker.selected;
            let current = model == &state.snapshot.model_name;
            lines.push(Line::from(vec![
                Span::styled(
                    if selected { " › " } else { "   " },
                    Style::default().fg(VY_VIOLET),
                ),
                Span::styled(
                    model.clone(),
                    Style::default()
                        .fg(if selected { VY_TEXT } else { VY_TECH_STRONG })
                        .add_modifier(if selected {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(
                    if current { "  current" } else { "" },
                    Style::default().fg(VY_SUCCESS),
                ),
            ]));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(" endpoint ", Style::default().fg(VY_DIM)),
        Span::styled(
            compact_path(&state.snapshot.base_url, 44),
            Style::default().fg(VY_TECH),
        ),
    ]));
    lines.push(Line::from(Span::styled(
        " ↑↓ select · Enter switch · Esc close",
        Style::default().fg(VY_DIM),
    )));
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .title(if layout.len < picker.models.len() {
                    format!(
                        " model profiles {}–{}/{} · click/Enter switch ",
                        layout.start + 1,
                        layout.start + layout.len,
                        picker.models.len()
                    )
                } else {
                    " model profiles · click/Enter switch ".to_string()
                })
                .borders(Borders::ALL)
                .border_style(Style::default().fg(VY_VIOLET))
                .style(Style::default().bg(VY_SURFACE)),
        ),
        popup,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelPickerLayout {
    area: Rect,
    start: usize,
    len: usize,
}

fn model_picker_layout(area: Rect, picker: &ModelPicker) -> ModelPickerLayout {
    let requested_height = u16::try_from(picker.models.len())
        .unwrap_or(u16::MAX)
        .saturating_add(5)
        .clamp(7, 16);
    let popup = centered(area, 56, requested_height);
    let capacity = usize::from(popup.height.saturating_sub(5));
    let len = picker.models.len().min(capacity);
    let start = list_window_start(picker.models.len(), len, picker.selected);
    ModelPickerLayout {
        area: popup,
        start,
        len,
    }
}

fn model_picker_index_at(layout: ModelPickerLayout, column: u16, row: u16) -> Option<usize> {
    let left = layout.area.x.saturating_add(1);
    let right = layout.area.right().saturating_sub(1);
    let offset = row.checked_sub(layout.area.y.saturating_add(1))? as usize;
    if column < left || column >= right || offset >= layout.len {
        return None;
    }
    Some(layout.start + offset)
}

fn list_window_start(total: usize, visible: usize, selected: usize) -> usize {
    if visible == 0 || total <= visible {
        return 0;
    }
    selected
        .min(total.saturating_sub(1))
        .saturating_sub(visible / 2)
        .min(total - visible)
}

fn render_clarification(frame: &mut ratatui::Frame<'_>, area: Rect, state: &UiState) {
    let Some(clarification) = &state.clarification else {
        return;
    };
    let question = clarification.current_question();
    let height = (question.options.len() as u16 * 2 + 9).clamp(10, 20);
    let popup = centered(area, 72, height);
    let header = question.header.as_deref().unwrap_or("Clarification");
    let mut lines = vec![
        Line::from(Span::styled(
            format!(
                "question {}/{} · {header}",
                clarification.question_index + 1,
                clarification.request.questions.len()
            ),
            Style::default().fg(VY_TECH),
        )),
        Line::from(""),
        Line::from(Span::styled(
            question.question.clone(),
            Style::default().fg(VY_TEXT).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
    ];
    if clarification.freeform {
        lines.push(Line::from(vec![
            Span::styled(" › ", Style::default().fg(VY_VIOLET)),
            Span::styled(
                if clarification.input.text.is_empty() {
                    "type your answer…".to_string()
                } else {
                    clarification.input.text.clone()
                },
                Style::default().fg(if clarification.input.text.is_empty() {
                    VY_DIM
                } else {
                    VY_TEXT
                }),
            ),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            " Enter submit · Esc choices",
            Style::default().fg(VY_DIM),
        )));
    } else {
        for (index, option) in question.options.iter().enumerate() {
            let selected = index == clarification.selected;
            lines.push(Line::from(vec![
                Span::styled(
                    if selected { " › " } else { "   " },
                    Style::default().fg(VY_VIOLET),
                ),
                Span::styled(
                    option.label.clone(),
                    Style::default()
                        .fg(if selected { VY_TEXT } else { VY_TECH_STRONG })
                        .add_modifier(if selected {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(
                    option
                        .description
                        .as_ref()
                        .map(|description| format!(" · {description}"))
                        .unwrap_or_default(),
                    Style::default().fg(VY_MUTED),
                ),
            ]));
        }
        let other_index = question.options.len();
        lines.push(Line::from(vec![
            Span::styled(
                if clarification.selected == other_index {
                    " › "
                } else {
                    "   "
                },
                Style::default().fg(VY_VIOLET),
            ),
            Span::styled("Other", Style::default().fg(VY_TECH_STRONG)),
            Span::styled(" · freeform answer", Style::default().fg(VY_MUTED)),
        ]));
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            " ↑↓ select · Enter answer · Esc cancel turn",
            Style::default().fg(VY_DIM),
        )));
    }
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: false }).block(
            Block::default()
                .title(" ask_user ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(VY_VIOLET))
                .style(Style::default().bg(VY_SURFACE_RAISED)),
        ),
        popup,
    );
    if clarification.freeform {
        let cursor_x = popup.x
            + 4
            + clarification
                .input
                .text
                .chars()
                .take(clarification.input.cursor)
                .count()
                .min(popup.width.saturating_sub(6) as usize) as u16;
        frame.set_cursor_position((cursor_x, popup.y + 5));
    }
}

fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width.saturating_sub(2)).max(1);
    let height = height.min(area.height.saturating_sub(2)).max(1);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn ratio_width(width: u16, value: usize, total: usize) -> u16 {
    ((width as usize).saturating_mul(value) / total.max(1)).min(width as usize) as u16
}

fn compact_path(path: &str, max: usize) -> String {
    if path.chars().count() <= max {
        return path.to_string();
    }
    format!(
        "…{}",
        path.chars()
            .rev()
            .take(max.saturating_sub(1))
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    )
}

fn format_number(value: isize) -> String {
    crate::tui::render::format_number(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn snapshot() -> ReplSnapshot {
        ReplSnapshot {
            cwd: "/tmp/vyrn".to_string(),
            model_name: "qwen-small".to_string(),
            base_url: "http://localhost:11434/v1".to_string(),
            debug_path: Some("/tmp/vyrn/.vyrn/debug/session.json".to_string()),
            context_used: 1009,
            context_limit: 4096,
            context_system: 244,
            context_history: 701,
            context_scratch: 64,
            turns: 2,
            session_spent: 4487,
            turn_saved: 128,
            session_saved: 512,
            manifest: "[machine] macos/arm64\n[env] cargo,git,rg".to_string(),
            skills: "skills: none".to_string(),
            stats: "session spent: 4487".to_string(),
            context: "context (estimated): 1009/4096".to_string(),
            scratchpad: "turn scratchpad: retained facts".to_string(),
            debug: "debug trace: /tmp/session.json".to_string(),
            models: vec!["qwen-small".to_string(), "llama-local".to_string()],
            prompt_history: Vec::new(),
        }
    }

    #[test]
    fn renders_the_mockup_regions_in_a_test_terminal() {
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = UiState::new(snapshot());
        state.open_inspector(InspectorKey::Scratchpad);
        terminal.draw(|frame| render(frame, &mut state)).unwrap();

        let text = buffer_text(terminal.backend().buffer());
        assert!(text.contains("VYRN"), "{text}");
        assert!(text.contains("cwd /tmp/vyrn"), "{text}");
        assert!(text.contains("qwen-small"), "{text}");
        assert!(text.contains("context 1,009 / 4,096"), "{text}");
        assert!(text.contains("inspect"), "{text}");
        assert!(text.contains("turn scratchpad: retained facts"), "{text}");
        assert!(text.contains("message vyrn"), "{text}");
        assert!(text.contains("session saved 512"), "{text}");
    }

    #[test]
    fn live_events_keep_tool_details_collapsed_until_clicked() {
        let mut state = UiState::new(snapshot());
        state.begin_turn(&UserTurnInput {
            text: "run the tests".to_string(),
            images: Vec::new(),
        });
        state.apply_update(TuiUpdate::ToolStarted {
            name: "batch".to_string(),
            input: r#"{"commands":["cargo test -q"]}"#.to_string(),
        });
        state.apply_update(TuiUpdate::ToolOk {
            name: "batch".to_string(),
            preview: "18 passed · 0 failed".to_string(),
        });
        state.apply_update(TuiUpdate::AssistantDelta(
            "18 tests passed. No failures.".to_string(),
        ));

        let text = transcript_lines(&state)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("tool.batch"), "{text}");
        assert!(text.contains("cargo test -q"), "{text}");
        assert!(!text.contains("18 passed · 0 failed"), "{text}");
        assert!(text.contains("18 tests passed. No failures."), "{text}");
        let tool_line = transcript_lines(&state)
            .into_iter()
            .map(|line| line.to_string())
            .find(|line| line.contains("tool.batch"))
            .unwrap();
        assert!(tool_line.starts_with("     ▸  "), "{tool_line}");

        state.toggle_tool_details(0, 0);
        let expanded = transcript_lines(&state)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(expanded.contains("18 passed · 0 failed"), "{expanded}");
        assert!(
            expanded
                .lines()
                .any(|line| line.starts_with("     ▾  tool.batch")),
            "{expanded}"
        );
    }

    #[test]
    fn every_interaction_keeps_its_response_counted_scratchpad() {
        let mut state = UiState::new(snapshot());
        for (index, tokens) in [17, 29].into_iter().enumerate() {
            state.begin_turn(&UserTurnInput {
                text: format!("interaction {}", index + 1),
                images: Vec::new(),
            });
            state.apply_update(TuiUpdate::ScratchpadDone {
                summary: Some(format!("retained fact {}", index + 1)),
                output_tokens: Some(TokenCount::provider(tokens)),
            });
            state.finish_turn(None);
        }
        state.begin_turn(&UserTurnInput {
            text: "interaction without tools".to_string(),
            images: Vec::new(),
        });
        state.finish_turn(None);

        let rows = transcript_rows(&state);
        assert!(rows.iter().any(|row| {
            row.target == Some(UiClickTarget::TurnScratchpad(0))
                && row.line.to_string().contains("17 provider tokens")
        }));
        assert!(rows.iter().any(|row| {
            row.target == Some(UiClickTarget::TurnScratchpad(1))
                && row.line.to_string().contains("29 provider tokens")
        }));
        assert!(rows.iter().any(|row| {
            row.target == Some(UiClickTarget::TurnScratchpad(2))
                && row.line.to_string().contains("scratchpad · none")
        }));

        state.open_turn_scratchpad(0);
        assert!(state.inspector_body.contains("retained fact 1"));
        assert!(state.inspector_body.contains("17 provider tokens"));
        state.open_turn_scratchpad(1);
        assert!(state.inspector_body.contains("retained fact 2"));
        assert!(state.inspector_body.contains("29 provider tokens"));
        state.open_turn_scratchpad(2);
        assert!(
            state
                .inspector_body
                .contains("interaction 3 scratchpad: none")
        );
    }

    #[test]
    fn rendered_interaction_disclosures_are_mouse_hit_targets() {
        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = UiState::new(snapshot());
        state.inspector = None;
        state.begin_turn(&UserTurnInput {
            text: "inspect this interaction".to_string(),
            images: Vec::new(),
        });
        state.apply_update(TuiUpdate::ToolStarted {
            name: "batch".to_string(),
            input: r#"{"commands":["date"]}"#.to_string(),
        });
        state.apply_update(TuiUpdate::ToolOk {
            name: "batch".to_string(),
            preview: "Sun Aug 23 17:40:47 WEST 2026".to_string(),
        });
        state.apply_update(TuiUpdate::ScratchpadDone {
            summary: Some("retained exact date".to_string()),
            output_tokens: Some(TokenCount::provider(18)),
        });
        state.finish_turn(None);
        terminal.draw(|frame| render(frame, &mut state)).unwrap();

        let tool = state
            .transcript_click_regions
            .iter()
            .find(|region| matches!(region.target, UiClickTarget::ToolDetails { .. }))
            .copied()
            .unwrap();
        let scratchpad = state
            .transcript_click_regions
            .iter()
            .find(|region| region.target == UiClickTarget::TurnScratchpad(0))
            .copied()
            .unwrap();
        assert_eq!(
            transcript_click_target(&state, tool.area.x, tool.area.y),
            Some(tool.target)
        );
        assert_eq!(
            transcript_click_target(&state, scratchpad.area.x, scratchpad.area.y),
            Some(scratchpad.target)
        );

        activate_local_click_target(tool.target, &mut state);
        assert!(matches!(
            &state.turns[0].events[0],
            TurnEvent::Tool(ToolCard { expanded: true, .. })
        ));
        activate_local_click_target(scratchpad.target, &mut state);
        assert_eq!(state.inspected_turn, Some(0));
        assert!(state.inspector_body.contains("retained exact date"));
    }

    #[test]
    fn transcript_keeps_breathing_room_within_and_between_interactions() {
        let mut state = UiState::new(snapshot());
        state.inspect_visible = false;
        for marker in ["FIRST", "SECOND"] {
            state.begin_turn(&UserTurnInput {
                text: format!("{marker}_USER"),
                images: Vec::new(),
            });
            state.apply_update(TuiUpdate::AssistantDelta(format!("{marker}_ANSWER")));
            state.finish_turn(None);
        }
        let lines = transcript_lines(&state)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let first_user = lines
            .iter()
            .position(|line| line.contains("FIRST_USER"))
            .unwrap();
        let first_answer = lines
            .iter()
            .position(|line| line.contains("FIRST_ANSWER"))
            .unwrap();
        let second_user = lines
            .iter()
            .position(|line| line.contains("SECOND_USER"))
            .unwrap();

        assert!(lines[first_user + 1].is_empty(), "{lines:?}");
        assert_eq!(first_answer + 3, second_user, "{lines:?}");
        assert!(
            lines[first_answer + 1..second_user]
                .iter()
                .all(String::is_empty)
        );
    }

    #[test]
    fn transcript_roles_use_distinct_full_row_backgrounds_and_shared_indentation() {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = UiState::new(snapshot());
        state.inspect_visible = false;
        state.begin_turn(&UserTurnInput {
            text: "USER_ROW_MARKER".to_string(),
            images: Vec::new(),
        });
        state.apply_update(TuiUpdate::AssistantDelta(
            "ASSISTANT_ROW_MARKER".to_string(),
        ));
        state.apply_update(TuiUpdate::AssistantDone);
        state.finish_turn(None);

        terminal.draw(|frame| render(frame, &mut state)).unwrap();
        let buffer = terminal.backend().buffer();
        let rows = buffer_text(buffer)
            .lines()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let user_y = rows
            .iter()
            .position(|row| row.contains("USER_ROW_MARKER"))
            .unwrap() as u16;
        let assistant_y = rows
            .iter()
            .position(|row| row.contains("ASSISTANT_ROW_MARKER"))
            .unwrap() as u16;

        let user_start = rows[user_y as usize].find("USER_ROW_MARKER").unwrap();
        assert_eq!(rows[user_y as usize][..user_start].chars().count(), 3);
        let assistant_start = rows[assistant_y as usize]
            .find("ASSISTANT_ROW_MARKER")
            .unwrap();
        assert_eq!(
            rows[assistant_y as usize][..assistant_start]
                .chars()
                .count(),
            3
        );
        assert_eq!(buffer[(0, user_y)].bg, VY_SURFACE_RAISED);
        assert_eq!(buffer[(79, user_y)].bg, VY_SURFACE_RAISED);
        assert_eq!(buffer[(0, assistant_y)].bg, VY_SURFACE);
        assert_eq!(buffer[(79, assistant_y)].bg, VY_SURFACE);
    }

    #[test]
    fn assistant_wait_uses_a_slow_thinking_animation() {
        let mut state = UiState::new(snapshot());
        state.begin_turn(&UserTurnInput {
            text: "hello".to_string(),
            images: Vec::new(),
        });
        state.apply_update(TuiUpdate::AssistantStart);

        assert_eq!(state.activity.as_deref(), Some("thinking"));
        assert_eq!(state.spinner_frame, 0);
        state.last_spinner_advance = Instant::now() - SPINNER_FRAME_INTERVAL;
        state.tick();
        assert_eq!(state.spinner_frame, 1);
        state.tick();
        assert_eq!(state.spinner_frame, 1);

        let text = transcript_lines(&state)
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("thinking"), "{text}");
    }

    #[test]
    fn command_palette_filters_and_accepts_a_command() {
        let mut state = UiState::new(snapshot());
        state.input.set("/scr".to_string());
        state.update_palette();
        assert!(state.palette_open);
        assert_eq!(state.palette_matches()[0].name, "/scratchpad");
        state.accept_palette();
        assert_eq!(state.input.text, "/scratchpad");
        assert!(!state.palette_open);
    }

    #[test]
    fn command_palette_hit_testing_tracks_terminal_resizes() {
        let mut state = UiState::new(snapshot());
        state.input.set("/".to_string());
        state.update_palette();

        for area in [Rect::new(0, 0, 120, 36), Rect::new(0, 0, 64, 18)] {
            let chunks = interface_chunks(area, &state);
            let palette = command_palette_area(chunks[5], state.palette_matches().len());
            assert!(palette.right() <= area.right());
            assert!(palette.bottom() <= area.bottom());
            assert_eq!(
                command_palette_index_at(
                    palette,
                    state.palette_matches().len(),
                    palette.x + 1,
                    palette.y + 1,
                ),
                Some(0)
            );
            assert_eq!(
                command_palette_index_at(
                    palette,
                    state.palette_matches().len(),
                    palette.right().saturating_sub(1),
                    palette.y + 1,
                ),
                None
            );
        }
    }

    #[test]
    fn header_and_inspect_strip_expose_clipped_resize_safe_click_regions() {
        let state = UiState::new(snapshot());
        for area in [
            Rect::new(0, 0, 120, 40),
            Rect::new(0, 0, 80, 24),
            Rect::new(0, 0, 64, 18),
        ] {
            let chunks = interface_chunks(area, &state);
            let header = header_click_regions(chunks[0], &state);
            let inspect = inspect_strip_click_regions(chunks[2]);
            assert!(
                header
                    .iter()
                    .all(|region| region.area.right() <= area.right())
            );
            assert!(
                inspect
                    .iter()
                    .all(|region| region.area.right() <= area.right()
                        && region.area.bottom() <= area.bottom())
            );

            let model = header
                .iter()
                .find(|region| region.target == UiClickTarget::OpenModels)
                .unwrap();
            assert_eq!(
                chrome_click_target(&chunks, &state, model.area.x, model.area.y),
                Some(UiClickTarget::OpenModels)
            );
            for target in [
                UiClickTarget::ToggleTrace,
                UiClickTarget::ToggleInspect,
                UiClickTarget::Clear,
            ] {
                let region = header
                    .iter()
                    .find(|region| region.target == target)
                    .unwrap();
                assert_eq!(
                    chrome_click_target(&chunks, &state, region.area.x, region.area.y),
                    Some(target)
                );
            }
        }

        let area = Rect::new(0, 0, 120, 40);
        let chunks = interface_chunks(area, &state);
        let header = header_click_regions(chunks[0], &state);
        for target in [
            UiClickTarget::ToggleTrace,
            UiClickTarget::ToggleInspect,
            UiClickTarget::Clear,
        ] {
            let region = header
                .iter()
                .find(|region| region.target == target)
                .unwrap();
            assert_eq!(
                chrome_click_target(&chunks, &state, region.area.x, region.area.y),
                Some(target)
            );
        }
        let inspect = inspect_strip_click_regions(chunks[2]);
        for target in [
            UiClickTarget::Inspector(InspectorKey::Stats),
            UiClickTarget::Refresh,
        ] {
            let region = inspect
                .iter()
                .find(|region| region.target == target)
                .unwrap();
            assert_eq!(
                chrome_click_target(&chunks, &state, region.area.x, region.area.y),
                Some(target)
            );
        }
        assert!(!inspect.iter().any(|region| {
            matches!(
                region.target,
                UiClickTarget::OpenModels | UiClickTarget::Inspector(InspectorKey::Debug)
            )
        }));
        assert_eq!(inspect.len(), 7);
    }

    #[test]
    fn inspect_buttons_stay_on_one_row_at_the_minimum_supported_width() {
        let state = UiState::new(snapshot());
        let area = Rect::new(0, 0, 64, 18);
        let chunks = interface_chunks(area, &state);
        let inspect = inspect_strip_click_regions(chunks[2]);

        assert_eq!(chunks[2].height, 3);
        assert_eq!(inspect.len(), 7);
        assert!(inspect.iter().all(|region| region.area.height == 3));
        assert!(inspect.windows(2).all(|regions| {
            regions[0].area.y == regions[1].area.y && regions[0].area.right() <= regions[1].area.x
        }));
        assert!(inspect.last().unwrap().area.right() <= area.right());
        assert_eq!(chunks[3].height, 0);
    }

    #[test]
    fn clicking_the_selected_inspector_toggles_its_panel() {
        let mut state = UiState::new(snapshot());

        toggle_inspector(&mut state, InspectorKey::Scratchpad);
        assert_eq!(state.inspector, None);
        assert_eq!(state.status, "scratchpad inspector collapsed");

        toggle_inspector(&mut state, InspectorKey::Scratchpad);
        assert_eq!(state.inspector, Some(InspectorKey::Scratchpad));
        assert!(state.inspect_visible);
        assert_eq!(state.status, "scratchpad inspector expanded");
    }

    #[test]
    fn active_turn_mouse_clicks_control_chrome_and_queue_mutations() {
        let area = Rect::new(0, 0, 120, 30);
        let mut state = UiState::new(snapshot());
        state.begin_turn(&UserTurnInput {
            text: "generate".to_string(),
            images: Vec::new(),
        });
        let (active_tx, mut active_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut deferred = DeferredActiveActions::default();

        let chunks = interface_chunks(area, &state);
        let scratchpad = inspect_strip_click_regions(chunks[2])
            .into_iter()
            .find(|region| region.target == UiClickTarget::Inspector(InspectorKey::Scratchpad))
            .unwrap();
        handle_active_mouse_in_area(
            left_mouse_down(scratchpad.area.x + 1, scratchpad.area.y + 1),
            area,
            &mut state,
            &active_tx,
            &mut deferred,
        );
        assert_eq!(state.inspector, None);

        let chunks = interface_chunks(area, &state);
        let models = header_click_regions(chunks[0], &state)
            .into_iter()
            .find(|region| region.target == UiClickTarget::OpenModels)
            .unwrap();
        handle_active_mouse_in_area(
            left_mouse_down(models.area.x, models.area.y),
            area,
            &mut state,
            &active_tx,
            &mut deferred,
        );
        let picker = model_picker_layout(area, state.model_picker.as_ref().unwrap());
        handle_active_mouse_in_area(
            left_mouse_down(picker.area.x + 1, picker.area.y + 2),
            area,
            &mut state,
            &active_tx,
            &mut deferred,
        );
        assert_eq!(deferred.model.as_deref(), Some("llama-local"));
        assert!(state.model_picker.is_none());

        let chunks = interface_chunks(area, &state);
        let clear = header_click_regions(chunks[0], &state)
            .into_iter()
            .find(|region| region.target == UiClickTarget::Clear)
            .unwrap();
        handle_active_mouse_in_area(
            left_mouse_down(clear.area.x, clear.area.y),
            area,
            &mut state,
            &active_tx,
            &mut deferred,
        );
        assert!(deferred.clear);
        assert!(matches!(active_rx.try_recv(), Ok(ActiveTurnInput::Cancel)));

        let refresh = inspect_strip_click_regions(chunks[2])
            .into_iter()
            .find(|region| region.target == UiClickTarget::Refresh)
            .unwrap();
        handle_active_mouse_in_area(
            left_mouse_down(refresh.area.x + 1, refresh.area.y + 1),
            area,
            &mut state,
            &active_tx,
            &mut deferred,
        );
        assert!(deferred.refresh_manifest);
    }

    #[test]
    fn model_picker_click_window_follows_selection_after_resize() {
        let mut state = UiState::new(snapshot());
        state.snapshot.models = (0..20).map(|index| format!("model-{index}")).collect();
        state.open_models();
        let picker = state.model_picker.as_mut().unwrap();
        picker.selected = picker.models.len() - 1;

        for area in [Rect::new(0, 0, 120, 40), Rect::new(0, 0, 64, 18)] {
            let layout = model_picker_layout(area, picker);
            assert!(layout.area.right() <= area.right());
            assert!(layout.area.bottom() <= area.bottom());
            assert_eq!(layout.start + layout.len, picker.models.len());
            assert_eq!(
                model_picker_index_at(layout, layout.area.x + 1, layout.area.y + layout.len as u16,),
                Some(picker.models.len() - 1)
            );
        }
    }

    fn buffer_text(buffer: &ratatui::buffer::Buffer) -> String {
        let area = buffer.area;
        (area.y..area.y + area.height)
            .map(|y| {
                (area.x..area.x + area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn left_mouse_down(column: u16, row: u16) -> crossterm::event::MouseEvent {
        crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(crossterm::event::MouseButton::Left),
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }
}
