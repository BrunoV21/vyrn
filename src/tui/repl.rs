use crate::agent::prompt::build_agent_prompt;
use crate::agent::tokens::{
    TokenBreakdown, TokenLedger, TurnUsage, estimate_assistant_output_tokens,
    estimate_chat_request_breakdown, estimate_messages_breakdown, estimate_unpruned_request_tokens,
};
use crate::agent::transcript::{Exchange, truncate};
use crate::agent::turn::{
    TurnScratchpad, build_fitted_turn_scratchpad_update_messages, build_turn_messages,
    prepare_next_turn_context, turn_scratchpad_update_source,
};
use crate::app::App;
use crate::config::{ModelProfile, ModelRegistry, ModelState};
use crate::debug_trace::{TraceMetadata, TraceRecorder};
use crate::llm::{
    ChatCompletionRequest, ChatMessage, ImageAttachment, LlmError, StreamEvent, ToolCall,
};
use crate::tools::{
    ASK_USER_TOOL_NAME, AskUserAnswer, AskUserRequest, AskUserResponse, MachineManifest, ToolResult,
};
use crate::vision;
use crossterm::cursor::{MoveToColumn, MoveUp, RestorePosition, SavePosition};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyModifiers,
};
use crossterm::execute;
use crossterm::style::{
    Attribute, Color, Print, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor,
    Stylize,
};
use crossterm::terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode, size};
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::IsTerminal;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, BufReader, Lines};

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
const MAX_TOOL_ROUNDS: usize = 64;
const MAX_PROMPT_HISTORY: usize = 100;
const BLOCK_SPACING_LINES: usize = 2;
const VY_VIOLET: Color = Color::Rgb {
    r: 139,
    g: 92,
    b: 246,
};
const VY_TECH: Color = Color::Rgb {
    r: 125,
    g: 162,
    b: 194,
};
const VY_TECH_STRONG: Color = Color::Rgb {
    r: 169,
    g: 189,
    b: 211,
};
const VY_SURFACE: Color = Color::Rgb {
    r: 13,
    g: 16,
    b: 22,
};
const VY_SURFACE_RAISED: Color = Color::Rgb {
    r: 21,
    g: 26,
    b: 36,
};
const VY_TEXT_MUTED: Color = Color::Rgb {
    r: 152,
    g: 163,
    b: 179,
};
const VY_TEXT_DIM: Color = Color::Rgb {
    r: 103,
    g: 114,
    b: 135,
};
const VY_SUCCESS: Color = Color::Rgb {
    r: 159,
    g: 232,
    b: 112,
};
const VY_RED: Color = Color::Rgb {
    r: 244,
    g: 63,
    b: 94,
};
const STEEL_BLUE: Color = VY_TECH;
const GRAPHITE_SURFACE_RAISED: Color = VY_SURFACE_RAISED;
const SYSTEM_SURFACE: Color = VY_SURFACE;

pub struct Repl {
    app: App,
    prompt_history: Vec<String>,
    plain_lines: Option<Lines<BufReader<tokio::io::Stdin>>>,
    input_pause: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Default)]
struct UserTurnInput {
    text: String,
    images: Vec<ImageAttachment>,
}

impl Repl {
    pub fn new(app: App) -> Self {
        let prompt_history = load_prompt_history(&app.sources);
        Self {
            app,
            prompt_history,
            plain_lines: None,
            input_pause: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        self.debug_log(format!(
            "session_start model={} base_url={} context={} verbose={}",
            self.app.model.name,
            self.app.model.base_url,
            self.app.config.context.max_tokens,
            self.app.verbose
        ));
        let result = if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
            self.run_inline_tui().await
        } else {
            self.run_plain().await
        };
        if let Some(trace) = self.app.trace.as_mut() {
            let _ = trace.finish("exit");
        }
        result
    }

    async fn run_plain(&mut self) -> anyhow::Result<()> {
        crate::tui::render::startup(
            &self.app.model.name,
            &self.app.model.base_url,
            &self.app.manifest,
            self.app.config.context.max_tokens,
        );

        self.plain_lines = Some(BufReader::new(tokio::io::stdin()).lines());

        loop {
            print!("you: ");
            std::io::stdout().flush()?;
            let Some(line) = self.next_plain_line().await? else {
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

            if let Err(error) = self
                .handle_user_turn(UserTurnInput {
                    text: input.to_string(),
                    images: Vec::new(),
                })
                .await
            {
                self.debug_log(format!("turn_error error={}", error));
                eprintln!("error: {}", format_error(&error, self.app.debug));
            }
        }

        self.plain_lines = None;
        Ok(())
    }

    async fn run_inline_tui(&mut self) -> anyhow::Result<()> {
        let _raw = RawModeGuard::enter()?;
        print_welcome(&self.app)?;
        let mut composer_status = self.composer_status_line();

        loop {
            let input = read_composer_line(&composer_status, &self.prompt_history)?;
            let input = UserTurnInput {
                text: input.text.trim().to_string(),
                images: input.images,
            };
            if input.text.is_empty() && input.images.is_empty() {
                continue;
            }
            if input.images.is_empty() && input.text.starts_with('/') {
                if self
                    .handle_inline_slash_command(&input.text, &mut composer_status)
                    .await?
                {
                    break;
                }
                continue;
            }
            self.remember_prompt(&input.text);
            let mut spinner: Option<Spinner> = None;
            let mut assistant_prefix_printed = false;
            let mut assistant_display_started = false;
            let mut assistant_renderer = MarkdownStreamRenderer::new();
            let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();
            let cancel_stop = Arc::new(AtomicBool::new(false));
            let cancel_handle = spawn_escape_listener(
                Arc::clone(&cancel_stop),
                Arc::clone(&self.input_pause),
                cancel_tx,
            );
            let turn = self.handle_user_turn_with(input, |update| match update {
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
                        let _ = assistant_renderer.push(&delta);
                    }
                }
                TuiUpdate::AssistantDone => {
                    if let Some(spinner) = spinner.take() {
                        spinner.stop();
                    }
                    if assistant_prefix_printed {
                        let _ = assistant_renderer.finish();
                        let _ = finish_assistant_block();
                    }
                }
                TuiUpdate::ToolStarted(name) => {
                    if let Some(spinner) = spinner.take() {
                        spinner.stop();
                    }
                    spinner = Some(Spinner::start(format!("running tool {name}")));
                }
                TuiUpdate::ToolInputStart => {
                    if let Some(spinner) = spinner.take() {
                        spinner.stop();
                    }
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
            });
            let result = tokio::select! {
                result = turn => result,
                _ = &mut cancel_rx => Err(LlmError::Canceled),
            };
            cancel_stop.store(true, Ordering::Relaxed);
            let _ = cancel_handle.join();
            if let Some(spinner) = spinner.take() {
                spinner.stop();
            }
            match result {
                Ok(()) => {}
                Err(LlmError::Canceled) => {
                    self.debug_log("turn_canceled");
                    composer_status = "last turn canceled".to_string();
                    let _ = print_system_block("canceled");
                }
                Err(error) => {
                    self.debug_log(format!("turn_error error={}", error));
                    let formatted = format_error(&error, self.app.debug);
                    composer_status = "last turn failed; run with --debug for details".to_string();
                    let _ = print_error_block(&formatted);
                }
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
                self.rotate_debug_trace("clear");
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

    async fn next_plain_line(&mut self) -> anyhow::Result<Option<String>> {
        let Some(lines) = self.plain_lines.as_mut() else {
            return Ok(None);
        };
        Ok(lines.next_line().await?)
    }

    async fn handle_inline_slash_command(
        &mut self,
        input: &str,
        composer_status: &mut String,
    ) -> anyhow::Result<bool> {
        match input {
            "/exit" => Ok(true),
            "/stats" => {
                print_stats_panel(
                    &self.app.stats,
                    self.current_context_tokens(),
                    self.app.config.context.max_tokens,
                    self.app.verbose,
                )?;
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
                self.rotate_debug_trace("clear");
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

    async fn handle_user_turn(&mut self, user_input: UserTurnInput) -> Result<(), LlmError> {
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
            TuiUpdate::ToolInputStart => {}
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
        user_input: UserTurnInput,
        mut emit: F,
    ) -> Result<(), LlmError>
    where
        F: FnMut(TuiUpdate),
    {
        let text_images = vision::attachments_from_text(&user_input.text)
            .await
            .map_err(|error| LlmError::Input(error.to_string()))?;
        let mut images = user_input.images;
        images.extend(text_images);
        dedupe_images(&mut images);
        images.truncate(vision::MAX_IMAGES_PER_MESSAGE);

        let initial_prompt = build_agent_prompt(
            &self.app.tools,
            &self.app.manifest,
            self.app.context.summary(),
            &user_input.text,
            &images,
        );
        self.debug_log(format!(
            "turn_start user_tokens={} images={} initial_prompt_tokens={} raw_history_tokens={} has_summary={}",
            crate::agent::tokens::estimate_text_tokens(&user_input.text),
            images.len(),
            initial_prompt.estimated_tokens.tokens,
            self.app.context.raw_history_tokens(),
            self.app.context.summary().is_some()
        ));
        let mut usage = TurnUsage::default();

        if self.app.context.previous_exchange().is_some() {
            emit(TuiUpdate::SummaryStart);
        }
        if let Some(summary_usage) = self
            .app
            .context
            .refresh_summary(
                &self.app.client,
                initial_prompt.estimated_tokens.tokens,
                self.app.trace.as_mut(),
                TraceMetadata {
                    action_type: "summary_refresh",
                    label: Some("summary".to_string()),
                    turn_index: Some(self.app.stats.turns.len()),
                    context_limit: Some(self.app.config.context.max_tokens),
                    ..TraceMetadata::default()
                },
            )
            .await?
        {
            let summary_total = summary_usage.input_tokens + summary_usage.output_tokens;
            self.debug_log(format!(
                "summary_refresh input_tokens={} output_tokens={} total_tokens={} next_prompt_estimate={}",
                summary_usage.input_tokens,
                summary_usage.output_tokens,
                summary_total,
                initial_prompt.estimated_tokens.tokens
            ));
            usage.add_call_with_breakdown(
                "summary",
                summary_total,
                summary_total,
                TokenBreakdown {
                    summary_inputs: summary_usage.input_tokens,
                    summary_outputs: summary_usage.output_tokens,
                    ..TokenBreakdown::default()
                },
            );
        }
        emit(TuiUpdate::SummaryDone);

        let prompt = build_agent_prompt(
            &self.app.tools,
            &self.app.manifest,
            self.app.context.summary(),
            &user_input.text,
            &images,
        );
        usage.context_tokens = prompt.estimated_tokens.tokens;
        let base_messages = prompt.messages;
        let mut scratchpad = TurnScratchpad::default();
        let mut current_tool_batch = Vec::new();
        let mut last_request_messages =
            build_turn_messages(&base_messages, &scratchpad, &current_tool_batch);
        let mut assistant_text = String::new();
        let mut all_tool_calls = Vec::new();
        let mut all_tool_results = Vec::new();
        let mut hit_tool_round_limit = false;

        for round in 0..MAX_TOOL_ROUNDS {
            let tool_schemas = self.app.tools.schemas();
            let messages = build_turn_messages(&base_messages, &scratchpad, &current_tool_batch);
            let request_breakdown = estimate_chat_request_breakdown(&messages, &tool_schemas);
            let request_tokens = request_breakdown.total();
            let request_would_be = estimate_unpruned_request_tokens(
                &request_breakdown,
                self.app.context.raw_history_tokens(),
            );
            self.debug_log(format!(
                "agent_request round={} request_tokens={} would_be={} max_context={} messages={} tool_schema_tokens={} summaries={} tool_outputs={} assistant_context={}",
                round,
                request_tokens,
                request_would_be,
                self.app.config.context.max_tokens,
                messages.len(),
                request_breakdown.tool_schemas,
                request_breakdown.summaries,
                request_breakdown.tool_call_outputs,
                request_breakdown.assistant_context
            ));
            let has_chained_context = round > 0
                || !scratchpad.summary.trim().is_empty()
                || !current_tool_batch.is_empty();
            if has_chained_context && request_tokens > self.app.config.context.max_tokens {
                return Err(LlmError::Input(format!(
                    "context budget exceeded before chained tool request: estimated {request_tokens} tokens exceeds configured {}",
                    self.app.config.context.max_tokens
                )));
            }

            emit(TuiUpdate::AssistantStart);
            let request = ChatCompletionRequest {
                model: String::new(),
                messages: messages.clone(),
                tools: tool_schemas.clone(),
                tool_choice: None,
                stream: true,
            };
            let pending_trace = self.app.trace.as_ref().map(|trace| {
                trace.begin_call(
                    &self.app.client,
                    &request,
                    true,
                    TraceMetadata {
                        action_type: "agent_turn",
                        label: Some(format!("agent-{round}")),
                        turn_index: Some(self.app.stats.turns.len()),
                        round_index: Some(round),
                        estimated_input_tokens: Some(request_tokens),
                        context_limit: Some(self.app.config.context.max_tokens),
                        token_breakdown: Some(request_breakdown),
                        ..TraceMetadata::default()
                    },
                )
            });
            let response_result = self
                .app
                .client
                .stream_chat(request, |event| match event {
                    StreamEvent::TextDelta(delta) => emit(TuiUpdate::AssistantDelta(delta)),
                    StreamEvent::ToolCallDone(call) => {
                        emit(TuiUpdate::ToolStarted(call.function.name));
                    }
                    StreamEvent::Finished => {}
                })
                .await;
            if let (Some(trace), Some(pending_trace)) = (self.app.trace.as_mut(), pending_trace) {
                let _ = trace.finish_call(pending_trace, &response_result);
            }
            let response = response_result?;
            emit(TuiUpdate::AssistantDone);
            last_request_messages = messages;

            let message = response
                .choices
                .first()
                .map(|choice| choice.message.clone())
                .ok_or(LlmError::MissingChoice)?;
            let output_tokens = response
                .usage
                .map(|usage| usage.completion_tokens)
                .filter(|tokens| *tokens > 0)
                .unwrap_or_else(|| estimate_assistant_output_tokens(&message));
            let mut call_breakdown = request_breakdown;
            call_breakdown.assistant_outputs += output_tokens;
            usage.add_call_with_breakdown(
                format!("agent-{round}"),
                request_tokens + output_tokens,
                request_would_be + output_tokens,
                call_breakdown,
            );

            if message.content.is_some() {
                if let Some(text) = message.content_text() {
                    assistant_text.push_str(text);
                } else {
                    assistant_text.push_str("[non-text assistant content]");
                }
            }

            let tool_calls = message.tool_calls.clone().unwrap_or_default();
            self.debug_log(format!(
                "agent_response round={} output_tokens={} tool_calls={} assistant_content={}",
                round,
                output_tokens,
                tool_calls.len(),
                message.content.is_some()
            ));
            if tool_calls.is_empty() {
                break;
            }

            let assistant_tool_message = message;
            current_tool_batch = vec![assistant_tool_message.clone()];
            for (tool_index, call) in tool_calls.into_iter().enumerate() {
                if call.function.name == ASK_USER_TOOL_NAME {
                    emit(TuiUpdate::ToolInputStart);
                }
                let result = self.execute_tool_call(&call).await;
                if matches!(result, Err(crate::tools::ToolError::Canceled)) {
                    return Err(LlmError::Canceled);
                }
                let mut incremental_batch = vec![assistant_tool_message.clone()];
                match &result {
                    Ok(tool_result) => {
                        emit(TuiUpdate::ToolOk {
                            name: tool_result.name.clone(),
                            preview: tool_preview(tool_result),
                        });
                        let tool_message = ChatMessage::tool(
                            call.id.clone(),
                            truncate(&tool_result.content, 8000),
                        );
                        incremental_batch.push(tool_message.clone());
                        current_tool_batch.push(tool_message);
                        all_tool_results.push(tool_result.clone());
                        let mut tool_images = tool_result.images.clone();
                        dedupe_images(&mut tool_images);
                        tool_images.truncate(vision::MAX_IMAGES_PER_MESSAGE);
                        if !tool_images.is_empty() {
                            let sources = tool_images
                                .iter()
                                .map(|image| image.source.as_str())
                                .collect::<Vec<_>>()
                                .join(", ");
                            let image_message = ChatMessage::user_with_images(
                                format!("Attached image(s) from read_image: {sources}"),
                                &tool_images,
                            );
                            incremental_batch.push(image_message.clone());
                            current_tool_batch.push(image_message);
                        }
                    }
                    Err(error) => {
                        let content = format!("tool error: {error}");
                        emit(TuiUpdate::ToolError {
                            name: call.function.name.clone(),
                            error: error.to_string(),
                        });
                        let tool_message = ChatMessage::tool(call.id.clone(), content);
                        incremental_batch.push(tool_message.clone());
                        current_tool_batch.push(tool_message);
                    }
                }
                all_tool_calls.push(call);
                scratchpad = self
                    .update_turn_scratchpad(
                        &scratchpad,
                        &incremental_batch,
                        &assistant_tool_message,
                        &mut usage,
                        self.app.config.context.max_tokens,
                        round,
                        tool_index,
                    )
                    .await?;
                self.debug_log(format!(
                    "turn_scratchpad_update round={} tool_index={} tokens={} chars={}",
                    round,
                    tool_index,
                    crate::agent::tokens::estimate_text_tokens(&scratchpad.summary),
                    scratchpad.summary.chars().count()
                ));
                let next_context = prepare_next_turn_context(
                    &base_messages,
                    &scratchpad,
                    &current_tool_batch,
                    &tool_schemas,
                    self.app.config.context.max_tokens,
                )?;
                scratchpad = next_context.scratchpad;
                current_tool_batch = next_context.tool_batch;
                let preparation = next_context.preparation;
                self.debug_log(format!(
                    "tool_chain_prepare round={} tool_index={} before_tokens={} after_tokens={} threshold={} max_context={} scratchpad_tokens={} current_batch_messages={}",
                    round,
                    tool_index,
                    preparation.before_tokens,
                    preparation.after_tokens,
                    preparation.threshold,
                    preparation.max_tokens,
                    crate::agent::tokens::estimate_text_tokens(&scratchpad.summary),
                    current_tool_batch.len()
                ));
            }
            if round + 1 == MAX_TOOL_ROUNDS {
                hit_tool_round_limit = true;
            }
        }

        self.app.context.set_previous_exchange(Exchange {
            user_input: exchange_user_input(&user_input.text, images.len()),
            assistant_text,
            tool_calls: all_tool_calls,
            tool_results: all_tool_results,
        });
        usage.context_tokens =
            estimate_chat_request_breakdown(&last_request_messages, &self.app.tools.schemas())
                .total();
        self.app.stats.push_turn(usage);
        if let Some(turn) = self.app.stats.turns.last() {
            self.debug_log(format!(
                "turn_complete sent={} would_be={} history_saved={} context_tokens={} session_sent={} session_would_be={} session_history_saved={}",
                turn.sent,
                turn.would_be,
                turn.saved,
                turn.context_tokens,
                self.app.stats.session_sent,
                self.app.stats.session_would_be,
                self.app.stats.session_saved
            ));
        }
        if hit_tool_round_limit {
            return Err(LlmError::ToolRoundLimit {
                rounds: MAX_TOOL_ROUNDS,
            });
        }
        emit(TuiUpdate::Stats(self.compact_stats_line()));
        if self.app.verbose
            && let Some(summary) = self.app.context.summary()
        {
            emit(TuiUpdate::Summary(summary.to_string()));
        }

        Ok(())
    }

    async fn update_turn_scratchpad(
        &mut self,
        scratchpad: &TurnScratchpad,
        consumed_tool_batch: &[ChatMessage],
        assistant_response: &ChatMessage,
        usage: &mut TurnUsage,
        max_tokens: usize,
        round: usize,
        tool_index: usize,
    ) -> Result<TurnScratchpad, LlmError> {
        let source = turn_scratchpad_update_source(consumed_tool_batch, assistant_response);
        let messages =
            build_fitted_turn_scratchpad_update_messages(&scratchpad.summary, &source, max_tokens);
        let estimated_input_tokens = estimate_messages_breakdown(&messages).total();
        self.debug_log(format!(
            "turn_scratchpad_request input_estimate={} source_chars={} previous_chars={}",
            estimated_input_tokens,
            messages
                .get(1)
                .and_then(ChatMessage::content_text)
                .map(|content| content.chars().count())
                .unwrap_or_default(),
            scratchpad.summary.chars().count()
        ));
        let request = ChatCompletionRequest {
            model: String::new(),
            messages,
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
        };
        let pending_trace = self.app.trace.as_ref().map(|trace| {
            trace.begin_call(
                &self.app.client,
                &request,
                false,
                TraceMetadata {
                    action_type: "turn_scratchpad",
                    label: Some("turn-scratchpad".to_string()),
                    turn_index: Some(self.app.stats.turns.len()),
                    round_index: Some(round),
                    tool_index: Some(tool_index),
                    estimated_input_tokens: Some(estimated_input_tokens),
                    context_limit: Some(max_tokens),
                    ..TraceMetadata::default()
                },
            )
        });
        let response_result = self.app.client.complete_chat(request).await;
        if let (Some(trace), Some(pending_trace)) = (self.app.trace.as_mut(), pending_trace) {
            let _ = trace.finish_call(pending_trace, &response_result);
        }
        let response = response_result?;
        let summary = response
            .choices
            .first()
            .and_then(|choice| choice.message.content_text().map(str::trim))
            .filter(|summary| !summary.is_empty())
            .ok_or_else(|| {
                LlmError::Input("turn scratchpad update returned an empty summary".to_string())
            })?
            .to_string();
        let usage_info = response.usage.unwrap_or_default();
        let input_tokens = if usage_info.prompt_tokens > 0 {
            usage_info.prompt_tokens
        } else {
            estimated_input_tokens
        };
        let output_tokens = if usage_info.completion_tokens > 0 {
            usage_info.completion_tokens
        } else {
            crate::agent::tokens::estimate_text_tokens(&summary)
        };
        usage.add_call_with_breakdown(
            "turn-scratchpad",
            input_tokens + output_tokens,
            input_tokens + output_tokens,
            TokenBreakdown {
                summary_inputs: input_tokens,
                summary_outputs: output_tokens,
                ..TokenBreakdown::default()
            },
        );
        self.debug_log(format!(
            "turn_scratchpad_response input_tokens={} output_tokens={} summary_tokens={}",
            input_tokens,
            output_tokens,
            crate::agent::tokens::estimate_text_tokens(&summary)
        ));
        Ok(TurnScratchpad { summary })
    }

    fn debug_log(&self, event: impl AsRef<str>) {
        if !self.app.debug {
            return;
        }
        let path = self.app.sources.project_vyrn.join("debug.log");
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
            return;
        };
        let _ = writeln!(file, "[{}] {}", unix_timestamp_millis(), event.as_ref());
    }

    fn rotate_debug_trace(&mut self, reason: &str) {
        let Some(trace) = self.app.trace.as_mut() else {
            return;
        };
        let _ = trace.finish(reason);
        self.app.trace = TraceRecorder::interactive(&self.app.sources, &self.app.client).ok();
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
        if call.function.name == ASK_USER_TOOL_NAME {
            return self.execute_ask_user(input).await;
        }
        let result = self.app.tools.execute(&call.function.name, input).await?;
        if result.refresh_manifest {
            self.refresh_manifest();
        }
        Ok(result)
    }

    async fn execute_ask_user(
        &mut self,
        input: Value,
    ) -> Result<ToolResult, crate::tools::ToolError> {
        let request = AskUserRequest::parse(input)?;
        let response = if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
            self.ask_user_inline(&request)?
        } else {
            self.ask_user_plain(&request).await?
        };
        response.into_tool_result()
    }

    async fn ask_user_plain(
        &mut self,
        request: &AskUserRequest,
    ) -> Result<AskUserResponse, crate::tools::ToolError> {
        let mut answers = Vec::with_capacity(request.questions.len());
        for (idx, question) in request.questions.iter().enumerate() {
            println!(
                "[ask_user {}/{}] {}",
                idx + 1,
                request.questions.len(),
                question
                    .header
                    .as_deref()
                    .filter(|header| !header.trim().is_empty())
                    .unwrap_or("clarification")
            );
            println!("{}", question.question);
            for (option_idx, option) in question.options.iter().enumerate() {
                if let Some(description) = option.description.as_deref() {
                    println!("{}. {} - {}", option_idx + 1, option.label, description);
                } else {
                    println!("{}. {}", option_idx + 1, option.label);
                }
            }
            if question.options.is_empty() {
                print!("reply: ");
            } else {
                println!("+. Other");
                print!("select option or type reply: ");
            }
            std::io::stdout().flush()?;
            let Some(line) =
                self.next_plain_line()
                    .await
                    .map_err(|error| crate::tools::ToolError::Failed {
                        tool: ASK_USER_TOOL_NAME.to_string(),
                        message: error.to_string(),
                    })?
            else {
                return Err(crate::tools::ToolError::Canceled);
            };
            let reply = line.trim().to_string();
            if let Ok(selected) = reply.parse::<usize>()
                && let Some(option) = question.options.get(selected.saturating_sub(1))
            {
                answers.push(AskUserAnswer::Option {
                    id: question.id.clone(),
                    answer: option.label.clone(),
                    option_index: selected.saturating_sub(1),
                    option_label: option.label.clone(),
                });
                continue;
            }
            answers.push(AskUserAnswer::Freeform {
                id: question.id.clone(),
                answer: reply,
            });
        }
        Ok(AskUserResponse { answers })
    }

    fn ask_user_inline(
        &mut self,
        request: &AskUserRequest,
    ) -> Result<AskUserResponse, crate::tools::ToolError> {
        let _pause = InputPauseGuard::new(&self.input_pause);
        let mut answers = Vec::with_capacity(request.questions.len());
        for (idx, question) in request.questions.iter().enumerate() {
            let answer = if question.options.is_empty() {
                let reply = read_clarification_freeform(question, idx, request.questions.len())?;
                AskUserAnswer::Freeform {
                    id: question.id.clone(),
                    answer: reply,
                }
            } else {
                match select_clarification_option(question, idx, request.questions.len())? {
                    ClarificationChoice::Option(option_index) => {
                        let option = &question.options[option_index];
                        AskUserAnswer::Option {
                            id: question.id.clone(),
                            answer: option.label.clone(),
                            option_index,
                            option_label: option.label.clone(),
                        }
                    }
                    ClarificationChoice::Other => {
                        let reply =
                            read_clarification_freeform(question, idx, request.questions.len())?;
                        AskUserAnswer::Freeform {
                            id: question.id.clone(),
                            answer: reply,
                        }
                    }
                }
            };
            answers.push(answer);
        }
        Ok(AskUserResponse { answers })
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

    fn remember_prompt(&mut self, input: &str) {
        let input = input.trim();
        if input.is_empty() || input.starts_with('/') {
            return;
        }
        if self
            .prompt_history
            .last()
            .is_some_and(|previous| previous == input)
        {
            return;
        }

        self.prompt_history.push(input.to_string());
        if self.prompt_history.len() > MAX_PROMPT_HISTORY {
            let to_drop = self.prompt_history.len() - MAX_PROMPT_HISTORY;
            self.prompt_history.drain(..to_drop);
        }
        let _ = save_prompt_history(&self.app.sources, &self.prompt_history);
    }

    fn full_stats_text(&self) -> String {
        let current_context = self.current_context_tokens();
        let mut text = format!(
            "session spent: {} | session raw-history would be: {} | session history saved: {} | context: {}/{}",
            self.app.stats.session_sent,
            self.app.stats.session_would_be,
            self.app.stats.session_saved,
            current_context,
            self.app.config.context.max_tokens,
        );
        if self.app.stats.session_sent > 0 {
            text.push_str(
                "\naccounting: spent is cumulative model I/O; history saved is prior raw history replaced by summaries; context is the final retained prompt.",
            );
        }
        if self.app.stats.session_sent > 0 {
            text.push_str("\ncontributors:");
            text.push_str(&format_breakdown(
                &self.session_breakdown(),
                self.app.stats.session_sent,
                8,
            ));
        }
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
                    text.push_str(&format_breakdown(&call.breakdown, call.sent, 4));
                }
            }
        }
        text
    }

    fn current_context_tokens(&self) -> usize {
        self.app
            .stats
            .turns
            .last()
            .map(|turn| turn.context_tokens)
            .unwrap_or_default()
    }

    fn session_breakdown(&self) -> TokenBreakdown {
        let mut breakdown = TokenBreakdown::default();
        for turn in &self.app.stats.turns {
            breakdown.add(turn.breakdown);
        }
        breakdown
    }

    fn compact_stats_line(&self) -> String {
        let Some(turn) = self.app.stats.turns.last() else {
            return self.composer_status_line();
        };
        format!(
            "turn spent: {} | turn history saved: {} | session history saved: {} | context: {}/{}",
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
            .map(|skill| skill.display_line())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn format_breakdown(breakdown: &TokenBreakdown, total: usize, limit: usize) -> String {
    let mut text = String::new();
    for item in breakdown.items().into_iter().take(limit) {
        let pct = if total == 0 {
            0
        } else {
            item.tokens.saturating_mul(100) / total
        };
        text.push_str(&format!(
            "\n  {}: {} ({}%)",
            item.label,
            crate::tui::render::format_number(item.tokens as isize),
            pct
        ));
    }
    text
}

fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

pub async fn select_model(models: &ModelRegistry) -> anyhow::Result<ModelProfile> {
    let profiles = models.list().cloned().collect::<Vec<_>>();
    if profiles.is_empty() {
        anyhow::bail!("no model profiles configured");
    }

    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        let _raw = RawModeGuard::enter()?;
        return select_model_with_arrows(&profiles)?
            .ok_or_else(|| anyhow::anyhow!("model selection canceled"));
    }

    select_model_by_number(&profiles)
}

fn select_model_by_number(profiles: &[ModelProfile]) -> anyhow::Result<ModelProfile> {
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

fn select_model_with_arrows(profiles: &[ModelProfile]) -> anyhow::Result<Option<ModelProfile>> {
    println!("\r\n{}", "models".with(STEEL_BLUE).bold());
    let mut selected = 0;
    render_model_picker(profiles, selected, false)?;

    loop {
        let Event::Key(key) = event::read()? else {
            continue;
        };
        match key.code {
            KeyCode::Esc => {
                println!("\r");
                return Ok(None);
            }
            KeyCode::Enter => {
                println!("\r");
                return Ok(profiles.get(selected).cloned());
            }
            KeyCode::Up => {
                selected = if selected == 0 {
                    profiles.len() - 1
                } else {
                    selected - 1
                };
                render_model_picker(profiles, selected, true)?;
            }
            KeyCode::Down => {
                selected = (selected + 1) % profiles.len();
                render_model_picker(profiles, selected, true)?;
            }
            KeyCode::Home => {
                selected = 0;
                render_model_picker(profiles, selected, true)?;
            }
            KeyCode::End => {
                selected = profiles.len() - 1;
                render_model_picker(profiles, selected, true)?;
            }
            _ => {}
        }
    }
}

fn render_model_picker(
    profiles: &[ModelProfile],
    selected: usize,
    redraw: bool,
) -> anyhow::Result<()> {
    let row_count = profiles.len().saturating_add(1);
    if redraw {
        execute!(
            std::io::stdout(),
            MoveUp(u16::try_from(row_count).unwrap_or(u16::MAX)),
            MoveToColumn(0)
        )?;
    }

    let (width, _) = size().unwrap_or((100, 24));
    let max_chars = usize::from(width).saturating_sub(4).max(1);

    for (idx, profile) in profiles.iter().enumerate() {
        let row = truncate_display(
            &format!(
                "{} ({}) @ {}",
                profile.name, profile.model, profile.base_url
            ),
            max_chars,
        );
        execute!(
            std::io::stdout(),
            MoveToColumn(0),
            Clear(ClearType::CurrentLine)
        )?;
        if idx == selected {
            execute!(
                std::io::stdout(),
                SetForegroundColor(STEEL_BLUE),
                Print("> "),
                Print(row),
                ResetColor,
                Print("\r\n")
            )?;
        } else {
            execute!(std::io::stdout(), Print("  "), Print(row), Print("\r\n"))?;
        }
    }

    let help = truncate_display(
        "Use Up/Down to choose, Enter to select, Esc to cancel.",
        max_chars,
    );
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(VY_TEXT_DIM),
        Print(help),
        ResetColor,
        Print("\r\n")
    )?;
    std::io::stdout().flush()?;
    Ok(())
}

#[derive(Debug, Clone)]
enum TuiUpdate {
    SummaryStart,
    SummaryDone,
    AssistantStart,
    AssistantDelta(String),
    AssistantDone,
    ToolStarted(String),
    ToolInputStart,
    ToolOk { name: String, preview: String },
    ToolError { name: String, error: String },
    Stats(String),
    Summary(String),
}

enum ClarificationChoice {
    Option(usize),
    Other,
}

struct InputPauseGuard {
    pause: Arc<AtomicBool>,
}

impl InputPauseGuard {
    fn new(pause: &Arc<AtomicBool>) -> Self {
        pause.store(true, Ordering::Relaxed);
        Self {
            pause: Arc::clone(pause),
        }
    }
}

impl Drop for InputPauseGuard {
    fn drop(&mut self) {
        self.pause.store(false, Ordering::Relaxed);
    }
}

struct RawModeGuard;

impl RawModeGuard {
    fn enter() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        execute!(std::io::stdout(), EnableBracketedPaste)?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = execute!(std::io::stdout(), DisableBracketedPaste);
        let _ = disable_raw_mode();
    }
}

fn spawn_escape_listener(
    stop: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    cancel_tx: tokio::sync::oneshot::Sender<()>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut cancel_tx = Some(cancel_tx);
        while !stop.load(Ordering::Relaxed) {
            if pause.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(50));
                continue;
            }
            match event::poll(Duration::from_millis(50)) {
                Ok(true) => {
                    let Ok(Event::Key(key)) = event::read() else {
                        continue;
                    };
                    if key.code == KeyCode::Esc {
                        if let Some(cancel_tx) = cancel_tx.take() {
                            let _ = cancel_tx.send(());
                        }
                        break;
                    }
                }
                Ok(false) => {}
                Err(_) => break,
            }
        }
    })
}

fn select_clarification_option(
    question: &crate::tools::AskUserQuestion,
    question_index: usize,
    question_count: usize,
) -> Result<ClarificationChoice, crate::tools::ToolError> {
    let mut selected = 0usize;
    render_clarification_picker(question, question_index, question_count, selected, false)?;
    let row_count = clarification_row_count(question);
    loop {
        let Event::Key(key) = event::read()? else {
            continue;
        };
        match key.code {
            KeyCode::Esc => {
                println!("\r");
                return Err(crate::tools::ToolError::Canceled);
            }
            KeyCode::Enter => {
                println!("\r");
                if selected < question.options.len() {
                    return Ok(ClarificationChoice::Option(selected));
                }
                return Ok(ClarificationChoice::Other);
            }
            KeyCode::Up => {
                selected = if selected == 0 {
                    row_count - 1
                } else {
                    selected - 1
                };
                render_clarification_picker(
                    question,
                    question_index,
                    question_count,
                    selected,
                    true,
                )?;
            }
            KeyCode::Down => {
                selected = (selected + 1) % row_count;
                render_clarification_picker(
                    question,
                    question_index,
                    question_count,
                    selected,
                    true,
                )?;
            }
            KeyCode::Home => {
                selected = 0;
                render_clarification_picker(
                    question,
                    question_index,
                    question_count,
                    selected,
                    true,
                )?;
            }
            KeyCode::End => {
                selected = row_count - 1;
                render_clarification_picker(
                    question,
                    question_index,
                    question_count,
                    selected,
                    true,
                )?;
            }
            _ => {}
        }
    }
}

fn read_clarification_freeform(
    question: &crate::tools::AskUserQuestion,
    question_index: usize,
    question_count: usize,
) -> Result<String, crate::tools::ToolError> {
    let mut input = String::new();
    render_clarification_freeform(question, question_index, question_count, &input, false)?;
    loop {
        let Event::Key(key) = event::read()? else {
            continue;
        };
        match key.code {
            KeyCode::Esc => {
                println!("\r");
                return Err(crate::tools::ToolError::Canceled);
            }
            KeyCode::Enter => {
                println!("\r");
                return Ok(input.trim().to_string());
            }
            KeyCode::Backspace => {
                input.pop();
                render_clarification_freeform(
                    question,
                    question_index,
                    question_count,
                    &input,
                    true,
                )?;
            }
            KeyCode::Char(ch) => {
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                    input.push(ch);
                    render_clarification_freeform(
                        question,
                        question_index,
                        question_count,
                        &input,
                        true,
                    )?;
                }
            }
            _ => {}
        }
    }
}

fn render_clarification_picker(
    question: &crate::tools::AskUserQuestion,
    question_index: usize,
    question_count: usize,
    selected: usize,
    redraw: bool,
) -> Result<(), crate::tools::ToolError> {
    if redraw {
        execute!(
            std::io::stdout(),
            MoveUp(u16::try_from(clarification_row_count(question) + 3).unwrap_or(u16::MAX)),
            MoveToColumn(0)
        )?;
    }
    render_clarification_header(question, question_index, question_count)?;
    let (width, _) = size().unwrap_or((100, 24));
    let max_chars = usize::from(width).saturating_sub(4).max(1);
    for (idx, option) in question.options.iter().enumerate() {
        let mut row = option.label.clone();
        if let Some(description) = option
            .description
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            row.push_str(" - ");
            row.push_str(description);
        }
        render_clarification_row(idx == selected, &truncate_display(&row, max_chars))?;
    }
    render_clarification_row(selected == question.options.len(), "+ Other...")?;
    render_clarification_help("Use Up/Down to choose, Enter to select, Esc to cancel.")?;
    Ok(())
}

fn render_clarification_freeform(
    question: &crate::tools::AskUserQuestion,
    question_index: usize,
    question_count: usize,
    input: &str,
    redraw: bool,
) -> Result<(), crate::tools::ToolError> {
    if redraw {
        execute!(std::io::stdout(), MoveUp(4), MoveToColumn(0))?;
    }
    render_clarification_header(question, question_index, question_count)?;
    let (width, _) = size().unwrap_or((100, 24));
    let max_chars = usize::from(width).saturating_sub(9).max(1);
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(VY_VIOLET),
        Print("reply: "),
        ResetColor,
        Print(truncate_display(input, max_chars)),
        Print("\r\n")
    )?;
    render_clarification_help("Type a reply, Enter to submit, Esc to cancel.")?;
    Ok(())
}

fn render_clarification_header(
    question: &crate::tools::AskUserQuestion,
    question_index: usize,
    question_count: usize,
) -> Result<(), crate::tools::ToolError> {
    let title = question
        .header
        .as_deref()
        .filter(|header| !header.trim().is_empty())
        .unwrap_or("clarification");
    let (width, _) = size().unwrap_or((100, 24));
    let max_chars = usize::from(width).saturating_sub(1).max(1);
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(STEEL_BLUE),
        Print(format!(
            "ask_user [{}/{}] {}",
            question_index + 1,
            question_count,
            truncate_display(title, max_chars.saturating_sub(18))
        )),
        ResetColor,
        Print("\r\n"),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        Print(truncate_display(&question.question, max_chars)),
        Print("\r\n")
    )?;
    Ok(())
}

fn render_clarification_row(selected: bool, text: &str) -> Result<(), crate::tools::ToolError> {
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine)
    )?;
    if selected {
        execute!(
            std::io::stdout(),
            SetForegroundColor(VY_VIOLET),
            Print("> "),
            Print(text),
            ResetColor,
            Print("\r\n")
        )?;
    } else {
        execute!(std::io::stdout(), Print("  "), Print(text), Print("\r\n"))?;
    }
    Ok(())
}

fn render_clarification_help(text: &str) -> Result<(), crate::tools::ToolError> {
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(VY_TEXT_DIM),
        Print(text),
        ResetColor,
        Print("\r\n")
    )?;
    std::io::stdout().flush()?;
    Ok(())
}

fn clarification_row_count(question: &crate::tools::AskUserQuestion) -> usize {
    question.options.len() + 1
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
                    SetForegroundColor(VY_TEXT_DIM),
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
    print_blank_lines(1)
}

fn print_block_spacer() -> anyhow::Result<()> {
    print_blank_lines(BLOCK_SPACING_LINES)
}

fn print_blank_lines(count: usize) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    execute!(stdout, ResetColor)?;
    for _ in 0..count {
        execute!(stdout, Print("\r\n"))?;
    }
    stdout.flush()?;
    Ok(())
}

fn print_stats_panel(
    ledger: &TokenLedger,
    current_context: usize,
    max_context: usize,
    verbose: bool,
) -> anyhow::Result<()> {
    print_stats_line(&[(String::from("stats"), VY_VIOLET)])?;

    print_stats_line(&[
        (String::from("session spent "), VY_TEXT_MUTED),
        (
            crate::tui::render::format_number(ledger.session_sent as isize),
            VY_TECH_STRONG,
        ),
        (String::from("  raw-history would be "), VY_TEXT_MUTED),
        (
            crate::tui::render::format_number(ledger.session_would_be as isize),
            VY_TECH_STRONG,
        ),
        (String::from("  history saved "), VY_TEXT_MUTED),
        (
            crate::tui::render::format_number(ledger.session_saved),
            if ledger.session_saved >= 0 {
                VY_SUCCESS
            } else {
                VY_RED
            },
        ),
        (String::from("  context "), VY_TEXT_MUTED),
        (
            format!(
                "{}/{}",
                crate::tui::render::format_number(current_context as isize),
                crate::tui::render::format_number(max_context as isize)
            ),
            VY_TECH,
        ),
    ])?;

    if ledger.session_sent == 0 {
        print_stats_line(&[(String::from("no completed requests yet"), VY_TEXT_DIM)])?;
        return print_spacer();
    }

    print_stats_line(&[(
        String::from(
            "spent is cumulative model I/O; saved is prior raw history replaced by summaries",
        ),
        VY_TEXT_DIM,
    )])?;
    print_stats_line(&[(
        String::from("context is the retained final prompt, not cumulative tokens sent"),
        VY_TEXT_DIM,
    )])?;

    print_stats_line(&[(String::from("contributors"), VY_VIOLET)])?;
    for item in session_breakdown(ledger).items().into_iter().take(8) {
        let pct = item.tokens.saturating_mul(100) / ledger.session_sent.max(1);
        let value = crate::tui::render::format_number(item.tokens as isize);
        print_stats_line(&[
            (String::from("  "), VY_TEXT_DIM),
            (item.label.to_string(), VY_TEXT_MUTED),
            (String::from(": "), VY_TEXT_DIM),
            (value, VY_TECH_STRONG),
            (String::from(" ("), VY_TEXT_DIM),
            (format!("{pct}%"), VY_TECH),
            (String::from(")"), VY_TEXT_DIM),
        ])?;
    }

    if verbose {
        print_stats_line(&[(String::from("turns"), VY_VIOLET)])?;
        for (idx, turn) in ledger.turns.iter().enumerate() {
            print_stats_line(&[
                (format!("  {}. ", idx + 1), VY_TEXT_DIM),
                (String::from("sent "), VY_TEXT_MUTED),
                (
                    crate::tui::render::format_number(turn.sent as isize),
                    VY_TECH_STRONG,
                ),
                (String::from("  would be "), VY_TEXT_MUTED),
                (
                    crate::tui::render::format_number(turn.would_be as isize),
                    VY_TECH_STRONG,
                ),
                (String::from("  saved "), VY_TEXT_MUTED),
                (
                    crate::tui::render::format_number(turn.saved),
                    if turn.saved >= 0 { VY_SUCCESS } else { VY_RED },
                ),
            ])?;
        }
    }

    print_spacer()
}

fn session_breakdown(ledger: &TokenLedger) -> TokenBreakdown {
    let mut breakdown = TokenBreakdown::default();
    for turn in &ledger.turns {
        breakdown.add(turn.breakdown);
    }
    breakdown
}

fn print_stats_line(segments: &[(String, Color)]) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    execute!(
        stdout,
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        ResetColor
    )?;
    for (text, color) in segments {
        execute!(stdout, SetForegroundColor(*color), Print(text))?;
    }
    execute!(stdout, ResetColor, Print("\r\n"))?;
    stdout.flush()?;
    Ok(())
}

fn read_composer_line(status: &str, history: &[String]) -> anyhow::Result<UserTurnInput> {
    let mut state = ComposerState::default();
    execute!(std::io::stdout(), SavePosition)?;
    render_composer_state(&state, status)?;

    loop {
        match event::read()? {
            Event::Key(key) => match handle_composer_key(key, &mut state, history)? {
                ComposerAction::Continue => {
                    render_composer_state(&state, status)?;
                }
                ComposerAction::Submit => {
                    clear_composer()?;
                    print_user_block(&state.input, state.images.len())?;
                    return Ok(UserTurnInput {
                        text: state.input,
                        images: state.images,
                    });
                }
                ComposerAction::Exit => {
                    return Ok(UserTurnInput {
                        text: "/exit".to_string(),
                        images: Vec::new(),
                    });
                }
            },
            Event::Paste(text) => {
                reset_history_navigation(&mut state);
                state.input.push_str(&text);
                reset_completion(&mut state.completion);
                render_composer_state(&state, status)?;
            }
            _ => {}
        }
    }
}

fn render_composer_state(state: &ComposerState, status: &str) -> anyhow::Result<()> {
    let completion_suffix = slash_completion_suffix(&state.input, &state.completion);
    render_composer(
        &state.input,
        state.images.len(),
        completion_suffix.as_deref(),
        status,
    )
}

#[derive(Default)]
struct ComposerState {
    input: String,
    images: Vec<ImageAttachment>,
    completion: CompletionState,
    history_cursor: Option<usize>,
    history_draft: String,
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
    state: &mut ComposerState,
    history: &[String],
) -> anyhow::Result<ComposerAction> {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            Ok(ComposerAction::Exit)
        }
        KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            Ok(ComposerAction::Exit)
        }
        KeyCode::Esc => Ok(ComposerAction::Exit),
        KeyCode::Enter => {
            accept_slash_completion(state);
            Ok(ComposerAction::Submit)
        }
        KeyCode::Backspace => {
            reset_history_navigation(state);
            state.input.pop();
            reset_completion(&mut state.completion);
            Ok(ComposerAction::Continue)
        }
        KeyCode::Up if key.modifiers.is_empty() => {
            history_previous(state, history);
            Ok(ComposerAction::Continue)
        }
        KeyCode::Down if key.modifiers.is_empty() => {
            history_next(state, history);
            Ok(ComposerAction::Continue)
        }
        KeyCode::Tab => {
            reset_history_navigation(state);
            accept_slash_completion(state);
            Ok(ComposerAction::Continue)
        }
        KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            reset_history_navigation(state);
            paste_from_clipboard(state);
            Ok(ComposerAction::Continue)
        }
        KeyCode::Char(ch) => {
            if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                reset_history_navigation(state);
                state.input.push(ch);
                reset_completion(&mut state.completion);
            }
            Ok(ComposerAction::Continue)
        }
        _ => Ok(ComposerAction::Continue),
    }
}

fn history_previous(state: &mut ComposerState, history: &[String]) {
    if history.is_empty() || !state.images.is_empty() {
        return;
    }
    if state.history_cursor.is_none() && state.input.starts_with('/') {
        return;
    }

    let cursor = match state.history_cursor {
        Some(0) => 0,
        Some(index) => index - 1,
        None => {
            state.history_draft = state.input.clone();
            history.len() - 1
        }
    };
    state.history_cursor = Some(cursor);
    state.input = history[cursor].clone();
    reset_completion(&mut state.completion);
}

fn history_next(state: &mut ComposerState, history: &[String]) {
    let Some(cursor) = state.history_cursor else {
        return;
    };

    if cursor + 1 < history.len() {
        let next = cursor + 1;
        state.history_cursor = Some(next);
        state.input = history[next].clone();
    } else {
        state.history_cursor = None;
        state.input = std::mem::take(&mut state.history_draft);
    }
    reset_completion(&mut state.completion);
}

fn reset_history_navigation(state: &mut ComposerState) {
    state.history_cursor = None;
    state.history_draft.clear();
}

fn load_prompt_history(sources: &crate::config::ConfigSources) -> Vec<String> {
    let path = sources.project_vyrn.join("history.json");
    let Ok(raw) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(mut history) = serde_json::from_str::<Vec<String>>(&raw) else {
        return Vec::new();
    };
    history.retain(|entry| {
        let trimmed = entry.trim();
        !trimmed.is_empty() && !trimmed.starts_with('/')
    });
    if history.len() > MAX_PROMPT_HISTORY {
        history.drain(..history.len() - MAX_PROMPT_HISTORY);
    }
    history
}

fn save_prompt_history(
    sources: &crate::config::ConfigSources,
    history: &[String],
) -> std::io::Result<()> {
    std::fs::create_dir_all(&sources.project_vyrn)?;
    let path = sources.project_vyrn.join("history.json");
    let raw = serde_json::to_string_pretty(history).unwrap_or_else(|_| "[]".to_string());
    std::fs::write(path, raw)
}

fn reset_completion(completion: &mut CompletionState) {
    completion.prefix.clear();
    completion.index = 0;
}

fn accept_slash_completion(state: &mut ComposerState) {
    let Some(command) = active_slash_completion(&state.input, &state.completion) else {
        return;
    };
    state.input = command.to_string();
    reset_completion(&mut state.completion);
}

fn slash_completion_suffix(input: &str, completion: &CompletionState) -> Option<String> {
    let command = active_slash_completion(input, completion)?;
    let suffix = command.strip_prefix(input)?;
    if suffix.is_empty() {
        None
    } else {
        Some(suffix.to_string())
    }
}

fn active_slash_completion(input: &str, completion: &CompletionState) -> Option<&'static str> {
    if let Some(command) = SLASH_COMMANDS
        .iter()
        .copied()
        .find(|command| *command == input)
    {
        return Some(command);
    }
    let matches = slash_completion_matches(input);
    if matches.is_empty() {
        return None;
    }
    let index = if completion.prefix == input {
        completion.index
    } else {
        0
    };
    matches.get(index % matches.len()).copied()
}

fn slash_completion_matches(input: &str) -> Vec<&'static str> {
    if !input.starts_with('/') || input.contains(' ') {
        return Vec::new();
    }
    SLASH_COMMANDS
        .iter()
        .copied()
        .filter(|command| command.starts_with(input))
        .collect::<Vec<_>>()
}

fn paste_from_clipboard(state: &mut ComposerState) {
    match vision::image_from_clipboard() {
        Ok(Some(image)) if state.images.len() < vision::MAX_IMAGES_PER_MESSAGE => {
            state.images.push(image);
            reset_completion(&mut state.completion);
        }
        Ok(Some(_)) => {}
        Ok(None) => {
            if let Ok(Some(text)) = vision::text_from_clipboard() {
                state.input.push_str(&text);
                reset_completion(&mut state.completion);
            }
        }
        Err(_) => {
            if let Ok(Some(text)) = vision::text_from_clipboard() {
                state.input.push_str(&text);
                reset_completion(&mut state.completion);
            }
        }
    }
}

fn render_composer(
    input: &str,
    image_count: usize,
    completion_suffix: Option<&str>,
    status: &str,
) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    let input_background = GRAPHITE_SURFACE_RAISED;
    execute!(
        stdout,
        RestorePosition,
        MoveToColumn(0),
        Clear(ClearType::FromCursorDown),
        SetBackgroundColor(input_background),
        Print(terminal_fill()),
        MoveToColumn(0),
        SetBackgroundColor(input_background),
        SetForegroundColor(STEEL_BLUE),
        Print("> "),
        SetBackgroundColor(input_background),
        SetForegroundColor(VY_TECH_STRONG),
        Print(input)
    )?;
    if image_count > 0 {
        execute!(
            stdout,
            SetBackgroundColor(input_background),
            SetForegroundColor(STEEL_BLUE),
            Print(format!(
                "  [{image_count} image{}]",
                if image_count == 1 { "" } else { "s" }
            ))
        )?;
    }
    if let Some(completion_suffix) = completion_suffix {
        execute!(
            stdout,
            SetBackgroundColor(input_background),
            SetForegroundColor(VY_TEXT_DIM),
            Print(completion_suffix)
        )?;
    }
    execute!(
        stdout,
        ResetColor,
        Print("\r\n\r\n"),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(VY_TEXT_DIM),
        Print(status),
        ResetColor,
        MoveUp(2),
        MoveToColumn((2 + input.chars().count()) as u16)
    )?;
    stdout.flush()?;
    Ok(())
}

fn clear_composer() -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        RestorePosition,
        MoveToColumn(0),
        Clear(ClearType::FromCursorDown)
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
    print_welcome_line(format!("+{border}+").with(STEEL_BLUE))?;
    print_welcome_line(banner_line(" __     __ __   __ ____  _   _ ", width).with(STEEL_BLUE))?;
    print_welcome_line(
        banner_line(" \\ \\   / / \\ \\ / /|  _ \\| \\ | |", width).with(STEEL_BLUE),
    )?;
    print_welcome_line(banner_line("  \\ \\ / /   \\ V / | |_) |  \\| |", width).with(STEEL_BLUE))?;
    print_welcome_line(banner_line("   \\ V /     | |  |  _ <| |\\  |", width).with(STEEL_BLUE))?;
    print_welcome_line(banner_line("    \\_/      |_|  |_| \\_\\_| \\_|", width).with(STEEL_BLUE))?;
    print_welcome_line(format!("+{border}+").with(STEEL_BLUE))?;
    print_welcome_line(format!(
        "{} {}  {}",
        "model".with(VY_TEXT_DIM),
        app.model.name.as_str().with(STEEL_BLUE),
        format!("context {}", app.config.context.max_tokens).with(VY_TEXT_DIM)
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

fn print_user_block(input: &str, image_count: usize) -> anyhow::Result<()> {
    let mut lines = user_display_lines(input, image_count).into_iter();
    if let Some(first_line) = lines.next() {
        print_block_line(
            ">",
            &first_line,
            GRAPHITE_SURFACE_RAISED,
            STEEL_BLUE,
            VY_TECH_STRONG,
        )?;
    }
    for line in lines {
        print_block_line(
            " ",
            &line,
            GRAPHITE_SURFACE_RAISED,
            STEEL_BLUE,
            VY_TECH_STRONG,
        )?;
    }
    print_block_spacer()
}

fn user_display_lines(input: &str, image_count: usize) -> Vec<String> {
    if image_count == 0 {
        split_terminal_lines(input)
    } else if input.trim().is_empty() {
        vec![format!(
            "[{image_count} image{} attached]",
            if image_count == 1 { "" } else { "s" }
        )]
    } else {
        let mut lines = split_terminal_lines(input);
        if let Some(last_line) = lines.last_mut() {
            last_line.push_str(&format!(
                "  [{} image{} attached]",
                image_count,
                if image_count == 1 { "" } else { "s" }
            ));
        }
        lines
    }
}

fn split_terminal_lines(text: &str) -> Vec<String> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let lines = normalized
        .split('\n')
        .map(str::to_string)
        .collect::<Vec<_>>();
    if lines.is_empty() {
        vec![String::new()]
    } else {
        lines
    }
}

fn print_assistant_prefix() -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(VY_TEXT_MUTED),
        Print("• "),
        SetForegroundColor(VY_TECH_STRONG),
    )?;
    std::io::stdout().flush()?;
    Ok(())
}

fn finish_assistant_block() -> anyhow::Result<()> {
    print_blank_lines(BLOCK_SPACING_LINES)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MarkdownStyle {
    bold: bool,
    italic: bool,
    strikethrough: bool,
    code: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StyledSegment {
    text: String,
    style: MarkdownStyle,
}

#[derive(Default)]
struct MarkdownStreamRenderer {
    pending: String,
    in_code_block: bool,
}

impl MarkdownStreamRenderer {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, text: &str) -> anyhow::Result<()> {
        self.pending.push_str(text);
        while let Some(newline) = self.pending.find('\n') {
            let mut line = self.pending.drain(..=newline).collect::<String>();
            if line.ends_with('\n') {
                line.pop();
            }
            if line.ends_with('\r') {
                line.pop();
            }
            self.print_line(&line)?;
            self.print_newline()?;
        }
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<()> {
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            self.print_line(line.trim_end_matches('\r'))?;
        }
        std::io::stdout().flush()?;
        Ok(())
    }

    fn print_line(&mut self, line: &str) -> anyhow::Result<()> {
        let segments = render_markdown_line(line, &mut self.in_code_block);
        print_styled_segments(&segments)
    }

    fn print_newline(&self) -> anyhow::Result<()> {
        execute!(
            std::io::stdout(),
            SetAttribute(Attribute::Reset),
            SetForegroundColor(VY_TECH_STRONG),
            Print("\r\n")
        )?;
        Ok(())
    }
}

fn render_markdown_line(line: &str, in_code_block: &mut bool) -> Vec<StyledSegment> {
    let trimmed = line.trim();
    if trimmed.starts_with("```") {
        *in_code_block = !*in_code_block;
        return Vec::new();
    }
    if *in_code_block {
        return vec![StyledSegment {
            text: line.to_string(),
            style: MarkdownStyle {
                code: true,
                ..Default::default()
            },
        }];
    }
    if is_markdown_rule(trimmed) {
        return vec![StyledSegment {
            text: "-".repeat(terminal_width().saturating_sub(2).min(72)),
            style: MarkdownStyle {
                strikethrough: true,
                ..Default::default()
            },
        }];
    }
    if let Some(heading) = strip_markdown_heading(line) {
        let mut segments = render_inline_markdown(heading, MarkdownStyle::default());
        for segment in &mut segments {
            segment.style.bold = true;
        }
        return segments;
    }
    render_inline_markdown(line, MarkdownStyle::default())
}

fn strip_markdown_heading(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let hash_count = trimmed.chars().take_while(|ch| *ch == '#').count();
    if !(1..=6).contains(&hash_count) {
        return None;
    }
    let after_hashes = &trimmed[hash_count..];
    if after_hashes.chars().next().is_some_and(char::is_whitespace) {
        Some(after_hashes.trim_start())
    } else {
        None
    }
}

fn is_markdown_rule(trimmed: &str) -> bool {
    if trimmed.len() < 3 {
        return false;
    }
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    matches!(first, '-' | '*' | '_') && chars.all(|ch| ch == first)
}

fn render_inline_markdown(input: &str, base_style: MarkdownStyle) -> Vec<StyledSegment> {
    let mut segments = Vec::new();
    let mut style = base_style;
    let mut index = 0;
    while index < input.len() {
        let rest = &input[index..];
        if let Some((marker, kind)) = markdown_marker(rest, style)
            && (marker_closes(kind, style) || has_closing_marker(&rest[marker.len()..], marker))
        {
            toggle_markdown_style(&mut style, kind);
            index += marker.len();
            continue;
        }
        if rest.starts_with('\\')
            && let Some((_, ch)) = rest.char_indices().nth(1)
        {
            push_styled_char(&mut segments, ch, style);
            index += 1 + ch.len_utf8();
            continue;
        }
        let Some(ch) = rest.chars().next() else {
            break;
        };
        push_styled_char(&mut segments, ch, style);
        index += ch.len_utf8();
    }
    segments
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarkdownMarker {
    Bold,
    Italic,
    Strikethrough,
    Code,
}

fn markdown_marker(rest: &str, style: MarkdownStyle) -> Option<(&'static str, MarkdownMarker)> {
    if style.code {
        return rest.starts_with('`').then_some(("`", MarkdownMarker::Code));
    }
    if rest.starts_with("**") {
        return Some(("**", MarkdownMarker::Bold));
    }
    if rest.starts_with("__") {
        return Some(("__", MarkdownMarker::Bold));
    }
    if rest.starts_with("~~") {
        return Some(("~~", MarkdownMarker::Strikethrough));
    }
    if rest.starts_with('`') {
        return Some(("`", MarkdownMarker::Code));
    }
    if !style.code && rest.starts_with('*') {
        return Some(("*", MarkdownMarker::Italic));
    }
    None
}

fn marker_closes(kind: MarkdownMarker, style: MarkdownStyle) -> bool {
    match kind {
        MarkdownMarker::Bold => style.bold,
        MarkdownMarker::Italic => style.italic,
        MarkdownMarker::Strikethrough => style.strikethrough,
        MarkdownMarker::Code => style.code,
    }
}

fn has_closing_marker(rest: &str, marker: &str) -> bool {
    !rest.chars().next().is_some_and(char::is_whitespace) && rest.contains(marker)
}

fn toggle_markdown_style(style: &mut MarkdownStyle, kind: MarkdownMarker) {
    match kind {
        MarkdownMarker::Bold => style.bold = !style.bold,
        MarkdownMarker::Italic => style.italic = !style.italic,
        MarkdownMarker::Strikethrough => style.strikethrough = !style.strikethrough,
        MarkdownMarker::Code => style.code = !style.code,
    }
}

fn push_styled_char(segments: &mut Vec<StyledSegment>, ch: char, style: MarkdownStyle) {
    if let Some(segment) = segments.last_mut()
        && segment.style == style
    {
        segment.text.push(ch);
        return;
    }
    segments.push(StyledSegment {
        text: ch.to_string(),
        style,
    });
}

fn print_styled_segments(segments: &[StyledSegment]) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    for segment in segments {
        execute!(
            stdout,
            SetAttribute(Attribute::Reset),
            SetForegroundColor(markdown_style_color(segment.style))
        )?;
        if segment.style.bold {
            execute!(stdout, SetAttribute(Attribute::Bold))?;
        }
        if segment.style.italic {
            execute!(stdout, SetAttribute(Attribute::Italic))?;
        }
        if segment.style.strikethrough {
            execute!(stdout, SetAttribute(Attribute::CrossedOut))?;
        }
        execute!(stdout, Print(&segment.text))?;
    }
    execute!(
        stdout,
        SetAttribute(Attribute::Reset),
        SetForegroundColor(VY_TECH_STRONG)
    )?;
    stdout.flush()?;
    Ok(())
}

fn markdown_style_color(style: MarkdownStyle) -> Color {
    if style.code {
        VY_SUCCESS
    } else if style.strikethrough {
        VY_TEXT_DIM
    } else {
        VY_TECH_STRONG
    }
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
            VY_SUCCESS,
            VY_TEXT_MUTED,
        ),
        ToolDisplayState::Failure => (
            Color::Rgb {
                r: 43,
                g: 12,
                b: 18,
            },
            VY_RED,
            VY_TEXT_MUTED,
        ),
    };
    print_block_line("tool", name, background, label_color, VY_TECH_STRONG)?;
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
        print_block_line("sys", line, SYSTEM_SURFACE, STEEL_BLUE, VY_TEXT_MUTED)?;
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
            VY_RED,
            VY_RED,
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

fn dedupe_images(images: &mut Vec<ImageAttachment>) {
    let mut seen = std::collections::BTreeSet::new();
    images.retain(|image| seen.insert(image.data_url()));
}

fn exchange_user_input(text: &str, image_count: usize) -> String {
    if image_count == 0 {
        text.to_string()
    } else if text.trim().is_empty() {
        format!("[attached images: {image_count}]")
    } else {
        format!("{text}\n[attached images: {image_count}]")
    }
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

    select_model_with_arrows(&profiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_history_moves_backward_and_forward() {
        let history = vec!["first prompt".to_string(), "second prompt".to_string()];
        let mut state = ComposerState {
            input: "draft".to_string(),
            ..Default::default()
        };

        history_previous(&mut state, &history);
        assert_eq!(state.input, "second prompt");
        history_previous(&mut state, &history);
        assert_eq!(state.input, "first prompt");
        history_next(&mut state, &history);
        assert_eq!(state.input, "second prompt");
        history_next(&mut state, &history);
        assert_eq!(state.input, "draft");
    }

    #[test]
    fn prompt_history_does_not_replace_slash_command_input() {
        let history = vec!["regular prompt".to_string()];
        let mut state = ComposerState {
            input: "/stats".to_string(),
            ..Default::default()
        };

        history_previous(&mut state, &history);
        assert_eq!(state.input, "/stats");
    }

    #[test]
    fn prompt_history_persists_in_project_vyrn() {
        let temp = tempfile::tempdir().unwrap();
        let sources = crate::config::ConfigSources::discover(temp.path().to_path_buf()).unwrap();
        let history = vec!["first prompt".to_string(), "second\nprompt".to_string()];

        save_prompt_history(&sources, &history).unwrap();

        assert_eq!(load_prompt_history(&sources), history);
        assert!(sources.project_vyrn.join("history.json").exists());
    }

    #[test]
    fn prompt_history_load_filters_commands_and_keeps_recent_entries() {
        let temp = tempfile::tempdir().unwrap();
        let sources = crate::config::ConfigSources::discover(temp.path().to_path_buf()).unwrap();
        std::fs::create_dir_all(&sources.project_vyrn).unwrap();
        let mut history = vec!["/stats".to_string(), "   ".to_string()];
        for index in 0..(MAX_PROMPT_HISTORY + 5) {
            history.push(format!("prompt {index}"));
        }
        std::fs::write(
            sources.project_vyrn.join("history.json"),
            serde_json::to_string(&history).unwrap(),
        )
        .unwrap();

        let loaded = load_prompt_history(&sources);

        assert_eq!(loaded.len(), MAX_PROMPT_HISTORY);
        assert_eq!(loaded.first().unwrap(), "prompt 5");
        assert_eq!(
            loaded.last().unwrap(),
            &format!("prompt {}", MAX_PROMPT_HISTORY + 4)
        );
    }

    #[test]
    fn slash_completion_suffix_shows_active_completion() {
        let state = ComposerState {
            input: "/sta".to_string(),
            ..Default::default()
        };

        assert_eq!(
            slash_completion_suffix(&state.input, &state.completion).as_deref(),
            Some("ts")
        );
    }

    #[test]
    fn slash_completion_does_not_extend_exact_command_alias() {
        let state = ComposerState {
            input: "/model".to_string(),
            ..Default::default()
        };

        assert_eq!(
            active_slash_completion(&state.input, &state.completion),
            Some("/model")
        );
        assert_eq!(
            slash_completion_suffix(&state.input, &state.completion),
            None
        );
    }

    #[test]
    fn tab_accepts_slash_completion_without_submitting() {
        let mut state = ComposerState {
            input: "/sta".to_string(),
            ..Default::default()
        };

        let action = handle_composer_key(
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
            &mut state,
            &[],
        )
        .unwrap();

        assert!(matches!(action, ComposerAction::Continue));
        assert_eq!(state.input, "/stats");
    }

    #[test]
    fn enter_submits_active_slash_completion() {
        let mut state = ComposerState {
            input: "/sta".to_string(),
            ..Default::default()
        };

        let action = handle_composer_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &mut state,
            &[],
        )
        .unwrap();

        assert!(matches!(action, ComposerAction::Submit));
        assert_eq!(state.input, "/stats");
    }

    #[test]
    fn user_display_lines_preserve_multiline_input() {
        assert_eq!(
            user_display_lines("first line\nsecond line", 0),
            vec!["first line".to_string(), "second line".to_string()]
        );
    }

    #[test]
    fn user_display_lines_attach_images_to_last_line() {
        assert_eq!(
            user_display_lines("first line\nsecond line", 2),
            vec![
                "first line".to_string(),
                "second line  [2 images attached]".to_string()
            ]
        );
    }

    #[test]
    fn markdown_line_strips_headings_and_inline_markers() {
        let mut in_code_block = false;
        let segments = render_markdown_line(
            "### 1. **Screen Analysis** with *detail* and `code`",
            &mut in_code_block,
        );

        let rendered = segment_text(&segments);
        assert_eq!(rendered, "1. Screen Analysis with detail and code");
        assert!(segments.iter().all(|segment| segment.style.bold));
        assert!(!rendered.contains('#'));
        assert!(!rendered.contains('*'));
        assert!(!rendered.contains('`'));
    }

    #[test]
    fn inline_markdown_maps_to_terminal_styles() {
        let segments = render_inline_markdown(
            "Use **bold**, *italic*, ~~old~~, and `src/lib.rs`.",
            MarkdownStyle::default(),
        );

        assert_segment_style(
            &segments,
            "bold",
            MarkdownStyle {
                bold: true,
                ..Default::default()
            },
        );
        assert_segment_style(
            &segments,
            "italic",
            MarkdownStyle {
                italic: true,
                ..Default::default()
            },
        );
        assert_segment_style(
            &segments,
            "old",
            MarkdownStyle {
                strikethrough: true,
                ..Default::default()
            },
        );
        assert_segment_style(
            &segments,
            "src/lib.rs",
            MarkdownStyle {
                code: true,
                ..Default::default()
            },
        );
        assert_eq!(
            segment_text(&segments),
            "Use bold, italic, old, and src/lib.rs."
        );
    }

    #[test]
    fn code_fences_are_hidden_and_code_lines_are_styled() {
        let mut in_code_block = false;
        assert!(render_markdown_line("```rust", &mut in_code_block).is_empty());
        assert!(in_code_block);

        let segments = render_markdown_line("let value = **literal**;", &mut in_code_block);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "let value = **literal**;");
        assert!(segments[0].style.code);

        assert!(render_markdown_line("```", &mut in_code_block).is_empty());
        assert!(!in_code_block);
    }

    fn segment_text(segments: &[StyledSegment]) -> String {
        segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect()
    }

    fn assert_segment_style(segments: &[StyledSegment], text: &str, style: MarkdownStyle) {
        let segment = segments
            .iter()
            .find(|segment| segment.text == text)
            .unwrap_or_else(|| panic!("missing segment {text:?}: {segments:?}"));
        assert_eq!(segment.style, style);
    }
}
