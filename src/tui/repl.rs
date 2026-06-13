use crate::agent::prompt::build_agent_prompt;
use crate::agent::tokens::{TurnUsage, estimate_messages_tokens};
use crate::agent::transcript::{Exchange, truncate};
use crate::app::App;
use crate::config::{ModelProfile, ModelRegistry, ModelState};
use crate::llm::{ChatCompletionRequest, ChatMessage, LlmError, StreamEvent, ToolCall};
use crate::tools::{MachineManifest, ToolResult};
use crossterm::cursor::{MoveDown, MoveToColumn, MoveUp};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor, Stylize};
use crossterm::terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size};
use serde_json::Value;
use std::io::IsTerminal;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};

const SLASH_COMMANDS: &[&str] = &[
    "/models",
    "/model",
    "/stats",
    "/manifest",
    "/refresh",
    "/skills",
    "/clear",
    "/exit",
];

pub struct Repl {
    app: App,
}

impl Repl {
    pub fn new(app: App) -> Self {
        Self { app }
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
            self.run_inline_tui().await
        } else {
            self.run_plain().await
        }
    }

    async fn run_plain(&mut self) -> anyhow::Result<()> {
        crate::tui::render::startup(
            &self.app.model.name,
            &self.app.model.base_url,
            &self.app.manifest,
            self.app.config.context.max_tokens,
        );

        let stdin = BufReader::new(tokio::io::stdin());
        let mut lines = stdin.lines();

        loop {
            print!("you: ");
            std::io::stdout().flush()?;
            let Some(line) = lines.next_line().await? else {
                break;
            };
            let input = line.trim();
            if input.is_empty() {
                continue;
            }
            if input.starts_with('/') {
                if self.handle_plain_slash_command(input).await? {
                    break;
                }
                continue;
            }

            if let Err(error) = self.handle_user_turn(input.to_string()).await {
                eprintln!("error: {}", format_error(&error, self.app.debug));
            }
        }

        Ok(())
    }

    async fn run_inline_tui(&mut self) -> anyhow::Result<()> {
        let _raw = RawModeGuard::enter()?;
        print_welcome(&self.app)?;
        let mut composer_status = self.composer_status_line();

        loop {
            let input = read_composer_line(&composer_status)?;
            let input = input.trim().to_string();
            if input.is_empty() {
                continue;
            }
            if input.starts_with('/') {
                if self
                    .handle_inline_slash_command(&input, &mut composer_status)
                    .await?
                {
                    break;
                }
                continue;
            }
            let mut spinner: Option<Spinner> = None;
            let mut assistant_prefix_printed = false;
            let mut assistant_display_started = false;
            let result = self
                .handle_user_turn_with(input, |update| match update {
                    TuiUpdate::SummaryStart => {
                        if let Some(spinner) = spinner.take() {
                            spinner.stop();
                        }
                        spinner = Some(Spinner::start("integrating previous turn"));
                    }
                    TuiUpdate::SummaryDone => {
                        if let Some(spinner) = spinner.take() {
                            spinner.stop();
                        }
                    }
                    TuiUpdate::AssistantStart => {
                        if let Some(spinner) = spinner.take() {
                            spinner.stop();
                        }
                        spinner = Some(Spinner::start("thinking"));
                    }
                    TuiUpdate::AssistantDelta(delta) => {
                        if let Some(spinner) = spinner.take() {
                            spinner.stop();
                        }
                        let delta = if assistant_display_started {
                            delta
                        } else {
                            delta.trim_start_matches(['\r', '\n']).to_string()
                        };
                        if !delta.is_empty() {
                            assistant_display_started = true;
                            if !assistant_prefix_printed {
                                let _ = print_assistant_prefix();
                                assistant_prefix_printed = true;
                            }
                            let _ = print_stream_text(&delta);
                        }
                    }
                    TuiUpdate::AssistantDone => {
                        if let Some(spinner) = spinner.take() {
                            spinner.stop();
                        }
                        if assistant_prefix_printed {
                            let _ = finish_assistant_block();
                        }
                    }
                    TuiUpdate::ToolStarted(name) => {
                        if let Some(spinner) = spinner.take() {
                            spinner.stop();
                        }
                        spinner = Some(Spinner::start(format!("running tool {name}")));
                    }
                    TuiUpdate::ToolOk { name, preview } => {
                        if let Some(spinner) = spinner.take() {
                            spinner.stop();
                        }
                        let _ = print_tool_preview(&name, &preview);
                    }
                    TuiUpdate::ToolError { name, error } => {
                        if let Some(spinner) = spinner.take() {
                            spinner.stop();
                        }
                        let _ = print_tool_error(&name, &error);
                    }
                    TuiUpdate::Stats(stats) => {
                        composer_status = stats;
                    }
                    TuiUpdate::Summary(summary) => {
                        let _ = print_system_block(&format!("summary\n{summary}"));
                    }
                })
                .await;
            if let Some(spinner) = spinner.take() {
                spinner.stop();
            }
            if let Err(error) = result {
                let formatted = format_error(&error, self.app.debug);
                composer_status = "last turn failed; run with --debug for details".to_string();
                let _ = print_error_block(&formatted);
            }
        }

        Ok(())
    }

    async fn handle_plain_slash_command(&mut self, input: &str) -> anyhow::Result<bool> {
        match input {
            "/exit" => Ok(true),
            "/stats" => {
                println!("{}", self.full_stats_text());
                Ok(false)
            }
            "/manifest" => {
                println!("{}", self.app.manifest.compact());
                Ok(false)
            }
            "/refresh" => {
                self.refresh_manifest();
                println!("manifest: {}", self.app.manifest.display_line());
                Ok(false)
            }
            "/skills" => {
                println!("{}", self.skills_text());
                Ok(false)
            }
            "/clear" => {
                self.app.context.clear();
                self.app.stats = Default::default();
                println!("cleared session context");
                Ok(false)
            }
            "/models" | "/model" => {
                let model = select_model(&self.app.models).await?;
                self.switch_model(model, true);
                Ok(false)
            }
            other => {
                println!("unknown command: {other}");
                Ok(false)
            }
        }
    }

    async fn handle_inline_slash_command(
        &mut self,
        input: &str,
        composer_status: &mut String,
    ) -> anyhow::Result<bool> {
        match input {
            "/exit" => Ok(true),
            "/stats" => {
                print_system_block(&self.full_stats_text())?;
                Ok(false)
            }
            "/manifest" => {
                print_system_block(&self.app.manifest.compact())?;
                Ok(false)
            }
            "/refresh" => {
                self.refresh_manifest();
                print_system_block(&format!("manifest: {}", self.app.manifest.display_line()))?;
                Ok(false)
            }
            "/skills" => {
                print_system_block(&self.skills_text())?;
                Ok(false)
            }
            "/clear" => {
                self.app.context.clear();
                self.app.stats = Default::default();
                *composer_status = self.composer_status_line();
                clear_screen()?;
                print_welcome(&self.app)?;
                Ok(false)
            }
            "/models" | "/model" => {
                if let Some(model) = select_model_inline(&self.app.models)? {
                    self.switch_model(model, true);
                    *composer_status = self.composer_status_line();
                    print_welcome(&self.app)?;
                }
                Ok(false)
            }
            other => {
                print_system_block(&format!("unknown command: {other}"))?;
                Ok(false)
            }
        }
    }

    async fn handle_user_turn(&mut self, user_input: String) -> Result<(), LlmError> {
        self.handle_user_turn_with(user_input, |update| match update {
            TuiUpdate::SummaryStart => println!("[integrating previous turn...]"),
            TuiUpdate::SummaryDone => {}
            TuiUpdate::AssistantDelta(delta) => {
                print!("{delta}");
                let _ = std::io::stdout().flush();
            }
            TuiUpdate::AssistantStart => {
                print!("vyrn: ");
                let _ = std::io::stdout().flush();
            }
            TuiUpdate::AssistantDone => println!(),
            TuiUpdate::ToolStarted(name) => println!("\n[tool {name}]"),
            TuiUpdate::ToolOk { name, preview } => {
                println!("[{name} ok]");
                if !preview.is_empty() {
                    println!("{preview}");
                }
            }
            TuiUpdate::ToolError { name, error } => println!("[{name} error] {error}"),
            TuiUpdate::Stats(stats) => println!("{stats}"),
            TuiUpdate::Summary(summary) => println!("[summary]\n{summary}"),
        })
        .await
    }

    async fn handle_user_turn_with<F>(
        &mut self,
        user_input: String,
        mut emit: F,
    ) -> Result<(), LlmError>
    where
        F: FnMut(TuiUpdate),
    {
        let initial_prompt = build_agent_prompt(
            &self.app.tools,
            &self.app.manifest,
            self.app.context.summary(),
            &user_input,
        );
        let mut usage = TurnUsage::default();

        if self.app.context.previous_exchange().is_some() {
            emit(TuiUpdate::SummaryStart);
        }
        if let Some(summary_sent) = self
            .app
            .context
            .refresh_summary(&self.app.client, initial_prompt.estimated_tokens.tokens)
            .await?
        {
            usage.add_call("summary", summary_sent, summary_sent);
        }
        emit(TuiUpdate::SummaryDone);

        let prompt = build_agent_prompt(
            &self.app.tools,
            &self.app.manifest,
            self.app.context.summary(),
            &user_input,
        );
        let would_be = self
            .app
            .context
            .estimate_would_be_tokens(&prompt.system, &user_input);
        usage.context_tokens = prompt.estimated_tokens.tokens;
        let mut messages = prompt.messages;
        let mut assistant_text = String::new();
        let mut all_tool_calls = Vec::new();
        let mut all_tool_results = Vec::new();

        for round in 0..8 {
            let sent = estimate_messages_tokens(&messages);
            usage.add_call(format!("agent-{round}"), sent, would_be);

            emit(TuiUpdate::AssistantStart);
            let response = self
                .app
                .client
                .stream_chat(
                    ChatCompletionRequest {
                        model: String::new(),
                        messages: messages.clone(),
                        tools: self.app.tools.schemas(),
                        tool_choice: None,
                        stream: true,
                    },
                    |event| match event {
                        StreamEvent::TextDelta(delta) => emit(TuiUpdate::AssistantDelta(delta)),
                        StreamEvent::ToolCallDone(call) => {
                            emit(TuiUpdate::ToolStarted(call.function.name));
                        }
                        StreamEvent::Finished => {}
                    },
                )
                .await?;
            emit(TuiUpdate::AssistantDone);

            let message = response
                .choices
                .first()
                .map(|choice| choice.message.clone())
                .ok_or(LlmError::MissingChoice)?;

            if let Some(content) = &message.content {
                assistant_text.push_str(content);
            }

            let tool_calls = message.tool_calls.clone().unwrap_or_default();
            if tool_calls.is_empty() {
                break;
            }

            messages.push(message);
            for call in tool_calls {
                let result = self.execute_tool_call(&call).await;
                match &result {
                    Ok(tool_result) => {
                        emit(TuiUpdate::ToolOk {
                            name: tool_result.name.clone(),
                            preview: tool_preview(tool_result),
                        });
                        messages.push(ChatMessage::tool(
                            call.id.clone(),
                            truncate(&tool_result.content, 8000),
                        ));
                        all_tool_results.push(tool_result.clone());
                    }
                    Err(error) => {
                        let content = format!("tool error: {error}");
                        emit(TuiUpdate::ToolError {
                            name: call.function.name.clone(),
                            error: error.to_string(),
                        });
                        messages.push(ChatMessage::tool(call.id.clone(), content));
                    }
                }
                all_tool_calls.push(call);
            }
        }

        self.app.context.set_previous_exchange(Exchange {
            user_input,
            assistant_text,
            tool_calls: all_tool_calls,
            tool_results: all_tool_results,
        });
        self.app.stats.push_turn(usage);
        emit(TuiUpdate::Stats(self.compact_stats_line()));
        if self.app.verbose
            && let Some(summary) = self.app.context.summary()
        {
            emit(TuiUpdate::Summary(summary.to_string()));
        }

        Ok(())
    }

    async fn execute_tool_call(
        &mut self,
        call: &ToolCall,
    ) -> Result<ToolResult, crate::tools::ToolError> {
        let input = if call.function.arguments.trim().is_empty() {
            Value::Object(Default::default())
        } else {
            serde_json::from_str(&call.function.arguments).map_err(|error| {
                crate::tools::ToolError::InvalidInput {
                    tool: call.function.name.clone(),
                    message: error.to_string(),
                }
            })?
        };
        let result = self.app.tools.execute(&call.function.name, input).await?;
        if result.refresh_manifest {
            self.refresh_manifest();
        }
        Ok(result)
    }

    fn refresh_manifest(&mut self) {
        self.app.manifest = MachineManifest::scan(&self.app.skills, &self.app.mcp);
    }

    fn switch_model(&mut self, model: ModelProfile, persist: bool) {
        self.app.client.switch_profile(model.clone());
        self.app.model = model;
        if persist {
            let _ = ModelState::save_last_selected(&self.app.sources, &self.app.model.name);
        }
    }

    fn full_stats_text(&self) -> String {
        let current_context = self
            .app
            .stats
            .turns
            .last()
            .map(|turn| turn.context_tokens)
            .unwrap_or_default();
        let mut text = format!(
            "session sent: {} | would_be: {} | saved: {} | context: {}/{}",
            self.app.stats.session_sent,
            self.app.stats.session_would_be,
            self.app.stats.session_saved,
            current_context,
            self.app.config.context.max_tokens,
        );
        if self.app.verbose {
            for (idx, turn) in self.app.stats.turns.iter().enumerate() {
                text.push_str(&format!(
                    "\nturn {} sent={} would_be={} saved={}",
                    idx + 1,
                    turn.sent,
                    turn.would_be,
                    turn.saved
                ));
                for call in &turn.calls {
                    text.push_str(&format!(
                        "\n  {} sent={} would_be={}",
                        call.label, call.sent, call.would_be
                    ));
                }
            }
        }
        text
    }

    fn compact_stats_line(&self) -> String {
        let Some(turn) = self.app.stats.turns.last() else {
            return self.composer_status_line();
        };
        format!(
            "tokens sent: {} | saved: {} | session saved: {} | context: {}/{}",
            crate::tui::render::format_number(turn.sent as isize),
            crate::tui::render::format_number(turn.saved),
            crate::tui::render::format_number(self.app.stats.session_saved),
            crate::tui::render::format_number(turn.context_tokens as isize),
            crate::tui::render::format_number(self.app.config.context.max_tokens as isize),
        )
    }

    fn composer_status_line(&self) -> String {
        format!(
            "{} | context: 0/{}",
            self.app.model.name,
            crate::tui::render::format_number(self.app.config.context.max_tokens as isize)
        )
    }

    fn skills_text(&self) -> String {
        if self.app.skills.is_empty() {
            return "skills: none".to_string();
        }
        self.app
            .skills
            .list()
            .map(|skill| format!("{} - {}", skill.name, skill.description))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub async fn select_model(models: &ModelRegistry) -> anyhow::Result<ModelProfile> {
    let profiles = models.list().cloned().collect::<Vec<_>>();
    if profiles.is_empty() {
        anyhow::bail!("no model profiles configured");
    }

    println!("configured models:");
    for (idx, profile) in profiles.iter().enumerate() {
        println!(
            "{}. {} ({}) @ {}",
            idx + 1,
            profile.name,
            profile.model,
            profile.base_url
        );
    }

    print!("select model [1]: ");
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let selected = input.trim().parse::<usize>().unwrap_or(1);
    let index = selected.saturating_sub(1);
    profiles
        .get(index)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("invalid model selection: {selected}"))
}

#[derive(Debug, Clone)]
enum TuiUpdate {
    SummaryStart,
    SummaryDone,
    AssistantStart,
    AssistantDelta(String),
    AssistantDone,
    ToolStarted(String),
    ToolOk { name: String, preview: String },
    ToolError { name: String, error: String },
    Stats(String),
    Summary(String),
}

struct RawModeGuard;

impl RawModeGuard {
    fn enter() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Spinner {
    fn start(label: impl Into<String>) -> Self {
        let label = label.into();
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let frames = ["|", "/", "-", "\\"];
            let mut idx = 0;
            let started = Instant::now();
            while !thread_stop.load(Ordering::Relaxed) {
                let elapsed = started.elapsed().as_secs().max(1);
                let _ = execute!(
                    std::io::stdout(),
                    MoveToColumn(0),
                    Clear(ClearType::CurrentLine),
                    SetForegroundColor(Color::DarkGrey),
                    Print(format!(
                        "{} Working ({}s • esc to interrupt) - {}",
                        frames[idx % frames.len()],
                        elapsed,
                        label
                    )),
                    ResetColor
                );
                let _ = std::io::stdout().flush();
                idx += 1;
                thread::sleep(Duration::from_millis(100));
            }
            let _ = execute!(
                std::io::stdout(),
                MoveToColumn(0),
                Clear(ClearType::CurrentLine)
            );
            let _ = std::io::stdout().flush();
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }

    fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn terminal_width() -> usize {
    size().map(|(width, _)| width as usize).unwrap_or(80)
}

fn terminal_fill() -> String {
    " ".repeat(terminal_width().saturating_sub(1))
}

fn print_block_line(
    label: &str,
    text: &str,
    background: Color,
    label_color: Color,
    text_color: Color,
) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    execute!(
        stdout,
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetBackgroundColor(background),
        Print(terminal_fill()),
        MoveToColumn(0),
        SetBackgroundColor(background),
        SetForegroundColor(label_color),
        Print(format!("{label} ")),
        SetBackgroundColor(background),
        SetForegroundColor(text_color),
        Print(text),
        ResetColor,
        Print("\r\n")
    )?;
    stdout.flush()?;
    Ok(())
}

fn print_spacer() -> anyhow::Result<()> {
    execute!(std::io::stdout(), ResetColor, Print("\r\n"))?;
    std::io::stdout().flush()?;
    Ok(())
}

fn read_composer_line(status: &str) -> anyhow::Result<String> {
    let mut input = String::new();
    let mut completion = CompletionState::default();
    render_composer(&input, None, status)?;

    loop {
        let Event::Key(key) = event::read()? else {
            continue;
        };
        match handle_composer_key(key, &mut input, &mut completion)? {
            ComposerAction::Continue => {
                let hints = slash_hints(&input);
                render_composer(&input, hints.as_deref(), status)?;
            }
            ComposerAction::Submit => {
                clear_composer()?;
                print_user_block(&input)?;
                return Ok(input);
            }
            ComposerAction::Exit => return Ok("/exit".to_string()),
        }
    }
}

#[derive(Default)]
struct CompletionState {
    prefix: String,
    index: usize,
}

enum ComposerAction {
    Continue,
    Submit,
    Exit,
}

fn handle_composer_key(
    key: KeyEvent,
    input: &mut String,
    completion: &mut CompletionState,
) -> anyhow::Result<ComposerAction> {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            Ok(ComposerAction::Exit)
        }
        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            Ok(ComposerAction::Exit)
        }
        KeyCode::Esc => Ok(ComposerAction::Exit),
        KeyCode::Enter => Ok(ComposerAction::Submit),
        KeyCode::Backspace => {
            input.pop();
            completion.prefix.clear();
            Ok(ComposerAction::Continue)
        }
        KeyCode::Tab => {
            autocomplete(input, completion);
            Ok(ComposerAction::Continue)
        }
        KeyCode::Char(ch) => {
            if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                input.push(ch);
                completion.prefix.clear();
            }
            Ok(ComposerAction::Continue)
        }
        _ => Ok(ComposerAction::Continue),
    }
}

fn autocomplete(input: &mut String, completion: &mut CompletionState) {
    if !input.starts_with('/') || input.contains(' ') {
        return;
    }
    let matches = SLASH_COMMANDS
        .iter()
        .copied()
        .filter(|command| command.starts_with(input.as_str()))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return;
    }

    if completion.prefix != *input {
        completion.prefix = input.clone();
        completion.index = 0;
    } else {
        completion.index = (completion.index + 1) % matches.len();
    }
    *input = matches[completion.index].to_string();
}

fn slash_hints(input: &str) -> Option<String> {
    if !input.starts_with('/') || input.contains(' ') {
        return None;
    }
    let matches = SLASH_COMMANDS
        .iter()
        .copied()
        .filter(|command| command.starts_with(input))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        None
    } else {
        Some(matches.join(" "))
    }
}

fn render_composer(input: &str, hints: Option<&str>, status: &str) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    let input_background = Color::Rgb {
        r: 28,
        g: 42,
        b: 60,
    };
    execute!(
        stdout,
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetBackgroundColor(input_background),
        Print(terminal_fill()),
        MoveToColumn(0),
        SetBackgroundColor(input_background),
        SetForegroundColor(Color::Cyan),
        Print("> "),
        SetBackgroundColor(input_background),
        SetForegroundColor(Color::White),
        Print(input)
    )?;
    if let Some(hints) = hints {
        execute!(
            stdout,
            SetBackgroundColor(input_background),
            SetForegroundColor(Color::DarkGrey),
            Print(format!("  {hints}"))
        )?;
    }
    execute!(
        stdout,
        ResetColor,
        Print("\r\n"),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(Color::DarkGrey),
        Print(status),
        ResetColor,
        MoveUp(1),
        MoveToColumn((2 + input.chars().count()) as u16)
    )?;
    stdout.flush()?;
    Ok(())
}

fn clear_composer() -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        MoveDown(1),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        MoveUp(1),
        MoveToColumn(0)
    )?;
    std::io::stdout().flush()?;
    Ok(())
}

fn clear_screen() -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        crossterm::cursor::MoveTo(0, 0),
        Clear(ClearType::All)
    )?;
    std::io::stdout().flush()?;
    Ok(())
}

fn print_welcome(app: &App) -> anyhow::Result<()> {
    let width = terminal_width().min(78).max(56);
    let border = "-".repeat(width.saturating_sub(2));
    print_welcome_line(format!("+{border}+").cyan())?;
    print_welcome_line(banner_line(" __     __ __   __ ____  _   _ ", width).cyan())?;
    print_welcome_line(banner_line(" \\ \\   / / \\ \\ / /|  _ \\| \\ | |", width).cyan())?;
    print_welcome_line(banner_line("  \\ \\ / /   \\ V / | |_) |  \\| |", width).cyan())?;
    print_welcome_line(banner_line("   \\ V /     | |  |  _ <| |\\  |", width).cyan())?;
    print_welcome_line(banner_line("    \\_/      |_|  |_| \\_\\_| \\_|", width).cyan())?;
    print_welcome_line(format!("+{border}+").cyan())?;
    print_welcome_line(format!(
        "{} {}  {}",
        "model".dark_grey(),
        app.model.name.as_str().cyan(),
        format!("context {}", app.config.context.max_tokens).dark_grey()
    ))?;
    execute!(std::io::stdout(), Print("\r\n"))?;
    std::io::stdout().flush()?;
    Ok(())
}

fn print_welcome_line(content: impl std::fmt::Display) -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Print(content),
        ResetColor,
        Print("\r\n")
    )?;
    Ok(())
}

fn banner_line(text: &str, width: usize) -> String {
    let inner_width = width.saturating_sub(4);
    let text = truncate_display(text, inner_width);
    format!("| {:inner_width$} |", text)
}

fn print_user_block(input: &str) -> anyhow::Result<()> {
    print_block_line(
        ">",
        input,
        Color::Rgb {
            r: 23,
            g: 35,
            b: 50,
        },
        Color::Grey,
        Color::White,
    )?;
    print_spacer()
}

fn print_assistant_prefix() -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetBackgroundColor(Color::Rgb { r: 8, g: 20, b: 34 }),
        Print(terminal_fill()),
        MoveToColumn(0),
        SetBackgroundColor(Color::Rgb { r: 8, g: 20, b: 34 }),
        SetForegroundColor(Color::Grey),
        Print("• "),
        SetBackgroundColor(Color::Rgb { r: 8, g: 20, b: 34 }),
        SetForegroundColor(Color::White),
    )?;
    std::io::stdout().flush()?;
    Ok(())
}

fn finish_assistant_block() -> anyhow::Result<()> {
    execute!(std::io::stdout(), ResetColor, Print("\r\n\r\n"))?;
    std::io::stdout().flush()?;
    Ok(())
}

fn print_stream_text(text: &str) -> anyhow::Result<()> {
    for ch in text.chars() {
        match ch {
            '\n' => print!("\r\n"),
            '\r' => {}
            other => print!("{other}"),
        }
    }
    std::io::stdout().flush()?;
    Ok(())
}

fn print_tool_preview(name: &str, preview: &str) -> anyhow::Result<()> {
    print_tool_block(name, preview, ToolDisplayState::Success)
}

fn print_tool_error(name: &str, error: &str) -> anyhow::Result<()> {
    print_tool_block(name, error, ToolDisplayState::Failure)
}

enum ToolDisplayState {
    Success,
    Failure,
}

fn print_tool_block(name: &str, body: &str, state: ToolDisplayState) -> anyhow::Result<()> {
    let (background, label_color, body_color) = match state {
        ToolDisplayState::Success => (
            Color::Rgb {
                r: 11,
                g: 38,
                b: 24,
            },
            Color::Green,
            Color::Grey,
        ),
        ToolDisplayState::Failure => (
            Color::Rgb {
                r: 43,
                g: 12,
                b: 18,
            },
            Color::Red,
            Color::Grey,
        ),
    };
    print_block_line("tool", name, background, label_color, Color::White)?;
    for line in body.lines().filter(|line| !line.trim().is_empty()).take(6) {
        print_block_line(
            "   ",
            &truncate_display(line, 120),
            background,
            label_color,
            body_color,
        )?;
    }
    print_spacer()
}

fn print_system_block(text: &str) -> anyhow::Result<()> {
    for line in text.lines() {
        print_block_line(
            "sys",
            line,
            Color::Rgb {
                r: 11,
                g: 19,
                b: 31,
            },
            Color::DarkGrey,
            Color::Grey,
        )?;
    }
    print_spacer()
}

fn print_error_block(text: &str) -> anyhow::Result<()> {
    for line in text.lines() {
        print_block_line(
            "error",
            line,
            Color::Rgb {
                r: 39,
                g: 12,
                b: 15,
            },
            Color::Red,
            Color::Red,
        )?;
    }
    print_spacer()
}

fn format_error(error: &LlmError, debug: bool) -> String {
    match error {
        LlmError::Request { url, source } => {
            let mut text = format!("network request failed while calling {url}");
            if debug {
                text.push_str(&format!("\nsource: {source:#}"));
                if source.is_timeout() {
                    text.push_str("\nkind: timeout");
                }
                if source.is_connect() {
                    text.push_str("\nkind: connection");
                }
                if source.is_decode() {
                    text.push_str("\nkind: decode");
                }
            } else {
                text.push_str(" (run with --debug for request details)");
            }
            text
        }
        LlmError::HttpStatus { url, status, body } => {
            let mut text = format!("provider returned HTTP {status} from {url}");
            if debug {
                if body.trim().is_empty() {
                    text.push_str("\nbody: <empty>");
                } else {
                    text.push_str("\nbody:\n");
                    text.push_str(body);
                }
            } else {
                text.push_str(" (run with --debug to show response body)");
            }
            text
        }
        other => {
            if debug {
                format!("{other:#?}")
            } else {
                other.to_string()
            }
        }
    }
}

fn tool_preview(result: &ToolResult) -> String {
    if result.name == "batch"
        && let Ok(commands) = serde_json::from_str::<Vec<Value>>(&result.content)
    {
        let mut lines = Vec::new();
        for command in commands.iter().take(3) {
            let command_text = command
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("<command>");
            let status = command
                .get("status")
                .map(|value| value.to_string())
                .unwrap_or_else(|| "timeout".to_string());
            lines.push(format!(
                "$ {}  status {}",
                truncate_display(command_text, 80),
                status
            ));
            if let Some(stdout) = command.get("stdout").and_then(Value::as_str)
                && let Some(line) = first_non_empty_line(stdout)
            {
                lines.push(format!("stdout: {}", truncate_display(line, 100)));
            }
            if let Some(stderr) = command.get("stderr").and_then(Value::as_str)
                && let Some(line) = first_non_empty_line(stderr)
            {
                lines.push(format!("stderr: {}", truncate_display(line, 100)));
            }
        }
        if commands.len() > 3 {
            lines.push(format!("... {} more command result(s)", commands.len() - 3));
        }
        return lines.join("\n");
    }

    result
        .content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(4)
        .map(|line| truncate_display(line, 120))
        .collect::<Vec<_>>()
        .join("\n")
}

fn first_non_empty_line(value: &str) -> Option<&str> {
    value.lines().find(|line| !line.trim().is_empty())
}

fn truncate_display(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

fn select_model_inline(models: &ModelRegistry) -> anyhow::Result<Option<ModelProfile>> {
    let profiles = models.list().cloned().collect::<Vec<_>>();
    if profiles.is_empty() {
        print_system_block("no model profiles configured")?;
        return Ok(None);
    }

    println!("\r\n{}", "models".cyan().bold());
    for (idx, profile) in profiles.iter().enumerate() {
        println!(
            "{} {} {} {}",
            format!("{}.", idx + 1).dark_grey(),
            profile.name.as_str().cyan(),
            profile.model.as_str().dark_grey(),
            profile.base_url.as_str().dark_grey()
        );
    }

    let mut selected = String::new();
    render_model_prompt(&selected)?;
    loop {
        let Event::Key(key) = event::read()? else {
            continue;
        };
        match key.code {
            KeyCode::Esc => {
                println!();
                return Ok(None);
            }
            KeyCode::Enter => {
                println!();
                let selected = selected.trim().parse::<usize>().unwrap_or(1);
                let index = selected.saturating_sub(1);
                return Ok(profiles.get(index).cloned());
            }
            KeyCode::Backspace => {
                selected.pop();
                render_model_prompt(&selected)?;
            }
            KeyCode::Char(ch) if ch.is_ascii_digit() => {
                selected.push(ch);
                render_model_prompt(&selected)?;
            }
            _ => {}
        }
    }
}

fn render_model_prompt(input: &str) -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(Color::Cyan),
        Print("select model [1]: "),
        ResetColor,
        Print(input)
    )?;
    std::io::stdout().flush()?;
    Ok(())
}
