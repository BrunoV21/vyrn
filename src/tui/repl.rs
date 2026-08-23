use crate::agent::prompt::build_agent_prompt;
use crate::agent::tokens::{
    TokenBreakdown, TokenCount, TokenLedger, TurnUsage, estimate_assistant_output_tokens,
    estimate_chat_request_breakdown, estimate_unpruned_request_tokens,
};
use crate::agent::transcript::{Exchange, truncate};
use crate::agent::turn::{
    TurnScratchpad, apply_live_steering_to_tool_batch, build_turn_messages, live_steering_message,
    prepare_next_turn_context, render_turn_scratchpad, update_turn_scratchpad,
};
use crate::app::App;
use crate::config::{ConfigSources, ModelProfile, ModelRegistry, ModelState};
use crate::debug_trace::{TraceMetadata, TraceRecorder};
use crate::llm::{
    ChatCompletionRequest, ChatMessage, ImageAttachment, LlmError, StreamEvent, StreamOptions,
    ToolCall,
};
use crate::tools::{
    ASK_USER_TOOL_NAME, AskUserAnswer, AskUserRequest, AskUserResponse, MachineManifest, ToolResult,
};
use crate::vision;
use crossterm::cursor::{MoveTo, MoveToColumn, MoveUp};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
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
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, BufReader, Lines};

#[derive(Debug, Clone, Copy)]
struct SlashCommand {
    name: &'static str,
    description: &'static str,
}

const SLASH_COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "/help",
        description: "list commands and keyboard controls",
    },
    SlashCommand {
        name: "/stats",
        description: "provider usage, estimates, and savings",
    },
    SlashCommand {
        name: "/context",
        description: "context used, available, and retained",
    },
    SlashCommand {
        name: "/summary",
        description: "show the current rolling summary",
    },
    SlashCommand {
        name: "/scratchpad",
        description: "show the last evolving turn scratchpad",
    },
    SlashCommand {
        name: "/models",
        description: "switch model profile (/model alias)",
    },
    SlashCommand {
        name: "/model",
        description: "alias for /models",
    },
    SlashCommand {
        name: "/manifest",
        description: "show the compact machine manifest",
    },
    SlashCommand {
        name: "/refresh",
        description: "rescan the machine manifest",
    },
    SlashCommand {
        name: "/skills",
        description: "list discovered skill sources",
    },
    SlashCommand {
        name: "/debug",
        description: "show debug trace status and path",
    },
    SlashCommand {
        name: "/clear",
        description: "reset context, scratchpad, and token stats",
    },
    SlashCommand {
        name: "/exit",
        description: "exit vyrn",
    },
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
    last_scratchpad: TurnScratchpad,
    last_scratchpad_tokens: TokenCount,
    prompt_history: Vec<String>,
    plain_lines: Option<Lines<BufReader<tokio::io::Stdin>>>,
    input_pause: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct UserTurnInput {
    pub text: String,
    pub images: Vec<ImageAttachment>,
}

#[derive(Debug, Clone)]
pub(super) struct ReplSnapshot {
    pub cwd: String,
    pub model_name: String,
    pub base_url: String,
    pub debug_path: Option<String>,
    pub context_used: usize,
    pub context_limit: usize,
    pub context_system: usize,
    pub context_history: usize,
    pub context_scratch: usize,
    pub turns: usize,
    pub session_spent: usize,
    pub turn_saved: isize,
    pub session_saved: isize,
    pub manifest: String,
    pub skills: String,
    pub stats: String,
    pub context: String,
    pub summary: String,
    pub scratchpad: String,
    pub debug: String,
    pub models: Vec<String>,
    pub prompt_history: Vec<String>,
}

impl Repl {
    pub fn new(app: App) -> Self {
        let prompt_history = load_prompt_history(&app.sources);
        Self {
            app,
            last_scratchpad: TurnScratchpad::default(),
            last_scratchpad_tokens: TokenCount::default(),
            prompt_history,
            plain_lines: None,
            input_pause: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn run(mut self) -> anyhow::Result<()> {
        let is_programmatic = self.app.prompt.is_some();
        self.debug_log(format!(
            "session_start model={} base_url={} context={} verbose={}",
            self.app.model.name,
            self.app.model.base_url,
            self.app.config.context.max_tokens,
            self.app.verbose
        ));
        let result = if let Some(prompt) = self.app.prompt.take() {
            self.run_programmatic(prompt).await
        } else if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
            self.run_inline_tui().await
        } else {
            self.run_plain().await
        };
        if let Some(trace) = self.app.trace.as_mut() {
            let reason = if is_programmatic {
                if result.is_ok() {
                    "prompt_complete"
                } else {
                    "prompt_error"
                }
            } else {
                "exit"
            };
            let _ = trace.finish(reason);
        }
        result
    }

    pub async fn run_fullscreen(mut self) -> anyhow::Result<()> {
        if self.app.prompt.is_some() {
            anyhow::bail!("vyrn tui is interactive and cannot be combined with --prompt");
        }
        self.debug_log(format!(
            "session_start interface=fullscreen model={} base_url={} context={} verbose={}",
            self.app.model.name,
            self.app.model.base_url,
            self.app.config.context.max_tokens,
            self.app.verbose
        ));
        let result = crate::tui::fullscreen::run(&mut self).await;
        if let Some(trace) = self.app.trace.as_mut() {
            let _ = trace.finish("exit");
        }
        result
    }

    async fn run_programmatic(&mut self, prompt: String) -> anyhow::Result<()> {
        let prompt = prompt.trim();
        if prompt.is_empty() {
            anyhow::bail!("--prompt requires non-empty text");
        }
        self.handle_user_turn(UserTurnInput {
            text: prompt.to_string(),
            images: Vec::new(),
        })
        .await?;
        Ok(())
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
            let (active_input_tx, mut active_input_rx) = tokio::sync::mpsc::unbounded_channel();
            let active_input_stop = Arc::new(AtomicBool::new(false));
            let active_input_buffer = Arc::new(Mutex::new(String::new()));
            let active_input_handle = spawn_active_turn_listener(
                Arc::clone(&active_input_stop),
                Arc::clone(&self.input_pause),
                Arc::clone(&active_input_buffer),
                active_input_tx,
            );
            let result = self
                .handle_user_turn_with_active_input(input, &mut active_input_rx, false, |update| {
                    match update {
                        TuiUpdate::SummaryStart => {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop();
                            }
                            spinner = Some(Spinner::start(
                                "integrating previous turn",
                                Arc::clone(&active_input_buffer),
                            ));
                        }
                        TuiUpdate::SummaryDone { .. } => {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop();
                            }
                        }
                        TuiUpdate::AssistantStart => {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop();
                            }
                            spinner =
                                Some(Spinner::start("thinking", Arc::clone(&active_input_buffer)));
                        }
                        TuiUpdate::AssistantDelta(delta) => {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop();
                            }
                            if active_input_buffer
                                .lock()
                                .is_ok_and(|input| !input.is_empty())
                            {
                                return;
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
                                assistant_prefix_printed = false;
                                assistant_display_started = false;
                                assistant_renderer = MarkdownStreamRenderer::new();
                            }
                        }
                        TuiUpdate::AssistantInterrupted => {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop();
                            }
                            assistant_prefix_printed = false;
                            assistant_display_started = false;
                            assistant_renderer = MarkdownStreamRenderer::new();
                            let _ = clear_active_turn_input();
                        }
                        TuiUpdate::ToolStarted { name, .. } => {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop();
                            }
                            spinner = Some(Spinner::start(
                                format!("running tool {name}"),
                                Arc::clone(&active_input_buffer),
                            ));
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
                        TuiUpdate::ScratchpadStart => {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop();
                            }
                            spinner = Some(Spinner::start(
                                "updating turn memory",
                                Arc::clone(&active_input_buffer),
                            ));
                        }
                        TuiUpdate::ScratchpadDone { .. } => {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop();
                            }
                        }
                        TuiUpdate::Steering(text) => {
                            if let Some(spinner) = spinner.take() {
                                spinner.stop();
                            }
                            let _ = print_steering_block(&text);
                        }
                        TuiUpdate::Stats(stats) => {
                            composer_status = stats;
                        }
                        TuiUpdate::Summary(summary) => {
                            let _ = print_system_block(&format!("summary\n{summary}"));
                        }
                        TuiUpdate::Clarification(_) => {}
                    }
                })
                .await;
            active_input_stop.store(true, Ordering::Relaxed);
            let _ = active_input_handle.join();
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
            "/help" => {
                println!("{}", help_text());
                Ok(false)
            }
            "/stats" => {
                println!("{}", self.full_stats_text());
                Ok(false)
            }
            "/context" => {
                println!("{}", self.context_text());
                Ok(false)
            }
            "/summary" => {
                println!("{}", self.rolling_summary_text());
                Ok(false)
            }
            "/scratchpad" => {
                println!("{}", self.scratchpad_text());
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
            "/debug" => {
                println!("{}", self.debug_status_text());
                Ok(false)
            }
            "/clear" => {
                self.app.context.clear();
                self.app.stats = Default::default();
                self.last_scratchpad = TurnScratchpad::default();
                self.last_scratchpad_tokens = TokenCount::default();
                self.rotate_debug_trace("clear");
                println!("cleared session context");
                Ok(false)
            }
            "/models" | "/model" => {
                let model = select_model(&self.app.sources, &mut self.app.models).await?;
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
            "/help" => {
                print_system_block(&help_text())?;
                Ok(false)
            }
            "/stats" => {
                print_stats_panel(
                    &self.app.stats,
                    self.current_context_tokens(),
                    self.app.config.context.max_tokens,
                    self.app.verbose,
                )?;
                Ok(false)
            }
            "/context" => {
                print_system_block(&self.context_text())?;
                Ok(false)
            }
            "/summary" => {
                print_system_block(&self.rolling_summary_text())?;
                Ok(false)
            }
            "/scratchpad" => {
                print_system_block(&self.scratchpad_text())?;
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
            "/debug" => {
                print_system_block(&self.debug_status_text())?;
                Ok(false)
            }
            "/clear" => {
                self.app.context.clear();
                self.app.stats = Default::default();
                self.last_scratchpad = TurnScratchpad::default();
                self.last_scratchpad_tokens = TokenCount::default();
                self.rotate_debug_trace("clear");
                *composer_status = self.composer_status_line();
                clear_screen()?;
                print_welcome(&self.app)?;
                Ok(false)
            }
            "/models" | "/model" => {
                if let Some(model) = select_model_inline(&self.app.sources, &mut self.app.models)? {
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
            TuiUpdate::SummaryStart => {
                println!("[integrating previous turn...] ");
                let _ = std::io::stdout().flush();
            }
            TuiUpdate::SummaryDone { .. } => {}
            TuiUpdate::AssistantDelta(delta) => {
                print!("{delta}");
                let _ = std::io::stdout().flush();
            }
            TuiUpdate::AssistantStart => {
                print!("vyrn: ");
                let _ = std::io::stdout().flush();
            }
            TuiUpdate::AssistantDone => println!(),
            TuiUpdate::AssistantInterrupted => println!(),
            TuiUpdate::ToolStarted { name, .. } => println!("\n[tool {name}]"),
            TuiUpdate::ToolInputStart => {}
            TuiUpdate::ToolOk { name, preview } => {
                println!("[{name} ok]");
                if !preview.is_empty() {
                    println!("{preview}");
                }
            }
            TuiUpdate::ToolError { name, error } => println!("[{name} error] {error}"),
            TuiUpdate::ScratchpadStart => {
                print!("[updating turn memory...] ");
                let _ = std::io::stdout().flush();
            }
            TuiUpdate::ScratchpadDone { .. } => println!(),
            TuiUpdate::Steering(text) => println!("[live steering] {text}"),
            TuiUpdate::Stats(stats) => println!("{stats}"),
            TuiUpdate::Summary(summary) => println!("[summary]\n{summary}"),
            TuiUpdate::Clarification(_) => {}
        })
        .await
    }

    async fn handle_user_turn_with<F>(
        &mut self,
        user_input: UserTurnInput,
        emit: F,
    ) -> Result<(), LlmError>
    where
        F: FnMut(TuiUpdate),
    {
        let (_active_input_tx, mut active_input_rx) = tokio::sync::mpsc::unbounded_channel();
        self.handle_user_turn_with_active_input(user_input, &mut active_input_rx, false, emit)
            .await
    }

    pub(super) async fn handle_user_turn_fullscreen<F>(
        &mut self,
        user_input: UserTurnInput,
        active_input: &mut tokio::sync::mpsc::UnboundedReceiver<ActiveTurnInput>,
        emit: F,
    ) -> Result<(), LlmError>
    where
        F: FnMut(TuiUpdate),
    {
        self.handle_user_turn_with_active_input(user_input, active_input, true, emit)
            .await
    }

    async fn handle_user_turn_with_active_input<F>(
        &mut self,
        user_input: UserTurnInput,
        active_input: &mut tokio::sync::mpsc::UnboundedReceiver<ActiveTurnInput>,
        fullscreen: bool,
        mut emit: F,
    ) -> Result<(), LlmError>
    where
        F: FnMut(TuiUpdate),
    {
        self.app.context.begin_turn(&user_input.text);
        let initial_memory = self.app.context.prompt_memory();
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
            initial_memory.as_deref(),
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
        let mut steering_inputs = Vec::new();

        if self.app.context.previous_exchange().is_some() {
            emit(TuiUpdate::SummaryStart);
        }
        let summary_or_input = {
            let summary_future = self.app.context.refresh_summary(
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
            );
            tokio::select! {
                biased;
                Some(input) = active_input.recv() => Err(input),
                summary = summary_future => Ok(summary),
            }
        };
        match summary_or_input {
            Ok(summary_result) => {
                if let Some(summary_usage) = summary_result? {
                    let summary_total = summary_usage.input_tokens + summary_usage.output_tokens;
                    self.debug_log(format!(
                        "summary_refresh input_tokens={} output_tokens={} total_tokens={} next_prompt_estimate={}",
                        summary_usage.input_tokens,
                        summary_usage.output_tokens,
                        summary_total,
                        initial_prompt.estimated_tokens.tokens
                    ));
                    usage.add_model_call_with_breakdown(
                        "summary",
                        TokenCount {
                            tokens: summary_usage.input_tokens,
                            source: summary_usage.input_source,
                        },
                        TokenCount {
                            tokens: summary_usage.output_tokens,
                            source: summary_usage.output_source,
                        },
                        summary_total,
                        TokenBreakdown {
                            summary_inputs: summary_usage.input_tokens,
                            summary_outputs: summary_usage.output_tokens,
                            ..TokenBreakdown::default()
                        },
                    );
                }
            }
            Err(ActiveTurnInput::Cancel) => return Err(LlmError::Canceled),
            Err(ActiveTurnInput::Steering(text)) => {
                self.debug_log("live_steering_summary_interrupt");
                emit(TuiUpdate::Steering(text.clone()));
                steering_inputs.push(text);
            }
            Err(ActiveTurnInput::Clarification(_)) => {
                return Err(LlmError::Input(
                    "received a clarification response without an active question".to_string(),
                ));
            }
        }
        let rolling_summary = self
            .app
            .context
            .summary()
            .filter(|summary| !summary.trim().is_empty())
            .map(str::to_string);
        let retained_tokens = rolling_summary.as_deref().map(|summary| {
            TokenCount::estimate(crate::agent::tokens::estimate_text_tokens(summary))
        });
        emit(TuiUpdate::SummaryDone {
            summary: rolling_summary,
            retained_tokens,
        });

        let prompt_memory = self.app.context.prompt_memory();
        let prompt = build_agent_prompt(
            &self.app.tools,
            &self.app.manifest,
            prompt_memory.as_deref(),
            &user_input.text,
            &images,
        );
        usage.context_tokens = prompt.estimated_tokens.tokens;
        let base_messages = prompt.messages;
        let mut scratchpad = TurnScratchpad::default();
        self.last_scratchpad = scratchpad.clone();
        self.last_scratchpad_tokens = TokenCount::default();
        let mut current_tool_batch = steering_inputs
            .iter()
            .map(|text| live_steering_message(text))
            .collect::<Vec<_>>();
        let mut last_request_messages =
            build_turn_messages(&base_messages, &scratchpad, &current_tool_batch);
        let mut assistant_text = String::new();
        let mut all_tool_calls = Vec::new();
        let mut all_tool_results = Vec::new();
        let mut hit_tool_round_limit = false;

        'agent_rounds: for round in 0..MAX_TOOL_ROUNDS {
            while let Ok(input) = active_input.try_recv() {
                match input {
                    ActiveTurnInput::Cancel => return Err(LlmError::Canceled),
                    ActiveTurnInput::Steering(text) => {
                        self.debug_log(format!("live_steering_queued round={round}"));
                        emit(TuiUpdate::Steering(text.clone()));
                        current_tool_batch.push(live_steering_message(&text));
                        steering_inputs.push(text);
                    }
                    ActiveTurnInput::Clarification(_) => {
                        return Err(LlmError::Input(
                            "received a clarification response without an active question"
                                .to_string(),
                        ));
                    }
                }
            }
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
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
                max_tokens: None,
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
            let mut streamed_text = String::new();
            let response_or_input = {
                let response_future = self.app.client.stream_chat(request, |event| match event {
                    StreamEvent::TextDelta(delta) => {
                        streamed_text.push_str(&delta);
                        emit(TuiUpdate::AssistantDelta(delta));
                    }
                    StreamEvent::ToolCallDone(call) => {
                        emit(TuiUpdate::ToolStarted {
                            name: call.function.name,
                            input: call.function.arguments,
                        });
                    }
                    StreamEvent::Finished => {}
                });
                tokio::select! {
                    biased;
                    Some(input) = active_input.recv() => Err(input),
                    response = response_future => Ok(response),
                }
            };
            let response = match response_or_input {
                Ok(response_result) => {
                    if let (Some(trace), Some(pending_trace)) =
                        (self.app.trace.as_mut(), pending_trace)
                    {
                        let _ = trace.finish_call(pending_trace, &response_result);
                    }
                    response_result?
                }
                Err(input) => {
                    let interrupted = Err(LlmError::Input(
                        "model request interrupted by live user steering".to_string(),
                    ));
                    if let (Some(trace), Some(pending_trace)) =
                        (self.app.trace.as_mut(), pending_trace)
                    {
                        let _ = trace.finish_call(pending_trace, &interrupted);
                    }
                    emit(TuiUpdate::AssistantInterrupted);
                    last_request_messages = messages;
                    let partial_output_tokens =
                        crate::agent::tokens::estimate_text_tokens(&streamed_text);
                    let mut interrupted_breakdown = request_breakdown;
                    interrupted_breakdown.assistant_outputs += partial_output_tokens;
                    usage.add_model_call_with_breakdown(
                        format!("agent-{round}-interrupted"),
                        TokenCount::estimate(request_tokens),
                        TokenCount::estimate(partial_output_tokens),
                        request_would_be.saturating_add(partial_output_tokens),
                        interrupted_breakdown,
                    );
                    match input {
                        ActiveTurnInput::Cancel => return Err(LlmError::Canceled),
                        ActiveTurnInput::Steering(text) => {
                            self.debug_log(format!(
                                "live_steering_model_interrupt round={round} partial_chars={}",
                                streamed_text.chars().count()
                            ));
                            emit(TuiUpdate::Steering(text.clone()));
                            current_tool_batch.push(live_steering_message(&text));
                            steering_inputs.push(text);
                            continue 'agent_rounds;
                        }
                        ActiveTurnInput::Clarification(_) => {
                            return Err(LlmError::Input(
                                "received a clarification response without an active question"
                                    .to_string(),
                            ));
                        }
                    }
                }
            };
            emit(TuiUpdate::AssistantDone);
            last_request_messages = messages;

            let message = response
                .choices
                .first()
                .map(|choice| choice.message.clone())
                .ok_or(LlmError::MissingChoice)?;
            let provider_usage = response.usage;
            let input = provider_usage
                .map(|usage| usage.prompt_tokens)
                .filter(|tokens| *tokens > 0)
                .map(TokenCount::provider)
                .unwrap_or_else(|| TokenCount::estimate(request_tokens));
            let output = provider_usage
                .map(|usage| usage.completion_tokens)
                .filter(|tokens| *tokens > 0)
                .map(TokenCount::provider)
                .unwrap_or_else(|| {
                    TokenCount::estimate(estimate_assistant_output_tokens(&message))
                });
            let output_tokens = output.tokens;
            let mut call_breakdown = request_breakdown;
            call_breakdown.assistant_outputs += output_tokens;
            let would_be = input
                .tokens
                .saturating_add(request_would_be.saturating_sub(request_tokens))
                .saturating_add(output.tokens);
            usage.add_model_call_with_breakdown(
                format!("agent-{round}"),
                input,
                output,
                would_be,
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
            let tool_count = tool_calls.len();
            current_tool_batch = vec![assistant_tool_message.clone()];
            for (call_index, call) in tool_calls.iter().cloned().enumerate() {
                all_tool_calls.push(call.clone());
                if call.function.name == ASK_USER_TOOL_NAME {
                    emit(TuiUpdate::ToolInputStart);
                }
                let result_or_input = if call.function.name == ASK_USER_TOOL_NAME && fullscreen {
                    Ok(self
                        .execute_ask_user_fullscreen(&call, active_input, &mut emit)
                        .await)
                } else if call.function.name == ASK_USER_TOOL_NAME {
                    Ok(self.execute_tool_call(&call).await)
                } else {
                    tokio::select! {
                        biased;
                        Some(input) = active_input.recv() => Err(input),
                        result = self.execute_tool_call(&call) => Ok(result),
                    }
                };
                let result = match result_or_input {
                    Ok(result) => result,
                    Err(ActiveTurnInput::Cancel) => return Err(LlmError::Canceled),
                    Err(ActiveTurnInput::Steering(text)) => {
                        self.debug_log(format!(
                            "live_steering_tool_interrupt round={round} tool={} index={call_index}",
                            call.function.name
                        ));
                        apply_live_steering_to_tool_batch(
                            &mut current_tool_batch,
                            &tool_calls[call_index..],
                            &text,
                        );
                        emit(TuiUpdate::Steering(text.clone()));
                        steering_inputs.push(text);
                        let next_context = prepare_next_turn_context(
                            &base_messages,
                            &scratchpad,
                            &current_tool_batch,
                            &tool_schemas,
                            self.app.config.context.max_tokens,
                        )?;
                        scratchpad = next_context.scratchpad;
                        current_tool_batch = next_context.tool_batch;
                        continue 'agent_rounds;
                    }
                    Err(ActiveTurnInput::Clarification(_)) => {
                        return Err(LlmError::Input(
                            "received a clarification response without an active question"
                                .to_string(),
                        ));
                    }
                };
                if matches!(result, Err(crate::tools::ToolError::Canceled)) {
                    return Err(LlmError::Canceled);
                }
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
                        current_tool_batch.push(tool_message);
                    }
                }
            }
            emit(TuiUpdate::ScratchpadStart);
            scratchpad = update_turn_scratchpad(&scratchpad, &current_tool_batch);
            let rendered_scratchpad = render_turn_scratchpad(&scratchpad);
            self.last_scratchpad_tokens = TokenCount::estimate(
                crate::agent::tokens::estimate_text_tokens(&rendered_scratchpad),
            );
            emit(TuiUpdate::ScratchpadDone {
                summary: Some(rendered_scratchpad.clone()),
                output_tokens: Some(self.last_scratchpad_tokens),
            });
            self.last_scratchpad = scratchpad.clone();
            self.debug_log(format!(
                "turn_scratchpad_update round={} tools={} tokens={} token_source={} chars={}",
                round,
                tool_count,
                self.last_scratchpad_tokens.tokens,
                self.last_scratchpad_tokens.source.label(),
                rendered_scratchpad.chars().count()
            ));
            let next_context = prepare_next_turn_context(
                &base_messages,
                &scratchpad,
                &current_tool_batch,
                &tool_schemas,
                self.app.config.context.max_tokens,
            )?;
            scratchpad = next_context.scratchpad;
            self.last_scratchpad = scratchpad.clone();
            let rendered_scratchpad = render_turn_scratchpad(&scratchpad);
            self.last_scratchpad_tokens = TokenCount::estimate(
                crate::agent::tokens::estimate_text_tokens(&rendered_scratchpad),
            );
            current_tool_batch = next_context.tool_batch;
            let preparation = next_context.preparation;
            self.debug_log(format!(
                "tool_chain_prepare round={} tools={} before_tokens={} after_tokens={} threshold={} max_context={} scratchpad_tokens={} current_batch_messages={}",
                round,
                tool_count,
                preparation.before_tokens,
                preparation.after_tokens,
                preparation.threshold,
                preparation.max_tokens,
                crate::agent::tokens::estimate_text_tokens(&rendered_scratchpad),
                current_tool_batch.len()
            ));
            if round + 1 == MAX_TOOL_ROUNDS {
                hit_tool_round_limit = true;
            }
        }

        self.app.context.set_previous_exchange(Exchange {
            user_input: exchange_user_input_with_steering(
                &user_input.text,
                images.len(),
                &steering_inputs,
            ),
            assistant_text,
            turn_scratchpad: render_turn_scratchpad(&scratchpad),
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

    async fn execute_ask_user_fullscreen<F>(
        &mut self,
        call: &ToolCall,
        active_input: &mut tokio::sync::mpsc::UnboundedReceiver<ActiveTurnInput>,
        emit: &mut F,
    ) -> Result<ToolResult, crate::tools::ToolError>
    where
        F: FnMut(TuiUpdate),
    {
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
        let request = AskUserRequest::parse(input)?;
        emit(TuiUpdate::Clarification(request));
        loop {
            match active_input.recv().await {
                Some(ActiveTurnInput::Clarification(response)) => {
                    return response.into_tool_result();
                }
                Some(ActiveTurnInput::Cancel) | None => {
                    return Err(crate::tools::ToolError::Canceled);
                }
                Some(ActiveTurnInput::Steering(_)) => {}
            }
        }
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
            "session spent: {} (provider: {}, estimated fallback: {}) | session raw-history would be (estimated): {} | history reduction (estimated): {} | memory overhead: {} | net vs raw history (estimated): {} | context (estimated): {}/{} | available: {}",
            self.app.stats.session_sent,
            self.app.stats.session_provider_tokens,
            self.app.stats.session_estimated_tokens,
            self.app.stats.session_would_be,
            self.app.stats.session_saved,
            self.app.stats.memory_overhead_tokens(),
            self.app.stats.net_history_savings(),
            current_context,
            self.app.config.context.max_tokens,
            self.app
                .config
                .context
                .max_tokens
                .saturating_sub(current_context),
        );
        if self.app.stats.session_sent > 0 {
            text.push_str(
                "\naccounting: provider counts are used when returned; fallback, context, would-be, and saved values are explicitly estimated.",
            );
        }
        if self.app.stats.session_sent > 0 {
            text.push_str("\ncontributors:");
            text.push_str(&format_breakdown(
                &self.session_breakdown(),
                self.app.stats.session_sent,
                16,
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
                        "\n  {} input={} ({}) output={} ({}) sent={} would_be_estimate={}",
                        call.label,
                        call.input_tokens,
                        call.input_source.label(),
                        call.output_tokens,
                        call.output_source.label(),
                        call.sent,
                        call.would_be,
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
            "{} | context: 0/{} | / or Ctrl+O: commands",
            self.app.model.name,
            crate::tui::render::format_number(self.app.config.context.max_tokens as isize)
        )
    }

    fn context_text(&self) -> String {
        let used = self.current_context_tokens();
        let limit = self.app.config.context.max_tokens;
        let available = limit.saturating_sub(used);
        let percent = used.saturating_mul(100) / limit.max(1);
        let rolling_summary = self
            .app
            .context
            .summary()
            .map(crate::agent::tokens::estimate_text_tokens)
            .unwrap_or_default();
        format!(
            "context (estimated): {used}/{limit} ({percent}%) | available: {available}\nrolling summary (estimated): {rolling_summary} | raw history (estimated): {}\nprovider-reported session tokens: {} | estimated fallback: {} | history reduction: {} | memory overhead: {} | net vs raw history: {}",
            self.app.context.raw_history_tokens(),
            self.app.stats.session_provider_tokens,
            self.app.stats.session_estimated_tokens,
            self.app.stats.session_saved,
            self.app.stats.memory_overhead_tokens(),
            self.app.stats.net_history_savings(),
        )
    }

    fn scratchpad_text(&self) -> String {
        let rendered_scratchpad = render_turn_scratchpad(&self.last_scratchpad);
        if rendered_scratchpad.is_empty() {
            "turn scratchpad: none (no tool-driven context has been compacted yet)".to_string()
        } else {
            format!(
                "turn scratchpad ({} estimated tokens, deterministic checkpoint):\n{}",
                self.last_scratchpad_tokens.tokens, rendered_scratchpad
            )
        }
    }

    fn rolling_summary_text(&self) -> String {
        let Some(summary) = self
            .app
            .context
            .summary()
            .filter(|summary| !summary.trim().is_empty())
        else {
            return "rolling summary: none (the first turn has no previous exchange to summarize)"
                .to_string();
        };
        format!(
            "rolling summary ({} estimated retained tokens; model-generated):\n{}",
            crate::agent::tokens::estimate_text_tokens(summary),
            summary.trim()
        )
    }

    fn debug_status_text(&self) -> String {
        match self.app.trace.as_ref() {
            Some(trace) => format!("debug trace: {}", trace.path().display()),
            None => "debug trace: off (start vyrn with --debug)".to_string(),
        }
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

    pub(super) fn fullscreen_snapshot(&self) -> ReplSnapshot {
        let context_used = self.current_context_tokens();
        let context_limit = self.app.config.context.max_tokens;
        let breakdown = self
            .app
            .stats
            .turns
            .last()
            .and_then(|turn| turn.calls.last())
            .map(|call| call.breakdown)
            .unwrap_or_default();
        let context_scratch = self.last_scratchpad_tokens.tokens.min(context_used);
        let context_system = breakdown
            .system_prompt
            .saturating_add(breakdown.skills)
            .saturating_add(breakdown.tool_schemas)
            .saturating_add(breakdown.overhead)
            .min(context_used.saturating_sub(context_scratch));
        let context_history = context_used
            .saturating_sub(context_system)
            .saturating_sub(context_scratch);
        let turn_saved = self
            .app
            .stats
            .turns
            .last()
            .map(|turn| turn.saved)
            .unwrap_or_default();

        ReplSnapshot {
            cwd: self.app.sources.project_root.display().to_string(),
            model_name: self.app.model.name.clone(),
            base_url: self.app.model.base_url.clone(),
            debug_path: self
                .app
                .trace
                .as_ref()
                .map(|trace| trace.path().display().to_string()),
            context_used,
            context_limit,
            context_system,
            context_history,
            context_scratch,
            turns: self.app.stats.turns.len(),
            session_spent: self.app.stats.session_sent,
            turn_saved,
            session_saved: self.app.stats.session_saved,
            manifest: self.app.manifest.compact(),
            skills: self.skills_text(),
            stats: self.full_stats_text(),
            context: self.context_text(),
            summary: self.rolling_summary_text(),
            scratchpad: self.scratchpad_text(),
            debug: self.debug_status_text(),
            models: self
                .app
                .models
                .list()
                .map(|model| model.name.clone())
                .collect(),
            prompt_history: self.prompt_history.clone(),
        }
    }

    pub(super) fn fullscreen_clear(&mut self) {
        self.app.context.clear();
        self.app.stats = Default::default();
        self.last_scratchpad = TurnScratchpad::default();
        self.last_scratchpad_tokens = TokenCount::default();
        self.rotate_debug_trace("clear");
    }

    pub(super) fn fullscreen_refresh_manifest(&mut self) {
        self.refresh_manifest();
    }

    pub(super) fn fullscreen_switch_model(&mut self, name: &str) -> bool {
        let Some(model) = self.app.models.get(name) else {
            return false;
        };
        self.switch_model(model, true);
        true
    }

    pub(super) fn fullscreen_remember_prompt(&mut self, input: &str) {
        self.remember_prompt(input);
    }

    pub(super) fn fullscreen_format_error(&self, error: &LlmError) -> String {
        format_error(error, self.app.debug)
    }
}

fn help_text() -> String {
    let mut lines = vec!["commands".to_string()];
    for command in SLASH_COMMANDS {
        lines.push(format!("  {:<12} {}", command.name, command.description));
        lines.push(String::new());
    }
    lines.push("controls".to_string());
    lines.push("  / or Ctrl+O  open command palette".to_string());
    lines.push("  Up/Down      select command or recall prompt".to_string());
    lines.push("  Left click   run a visible palette command".to_string());
    lines.push("  Mouse wheel  move through palette commands".to_string());
    lines.push("  Tab           accept selected command".to_string());
    lines.push("  type + Enter  steer the agent during an active turn".to_string());
    lines.push("  Esc           cancel active turn; exit from composer".to_string());
    lines.join("\n")
}

fn format_breakdown(breakdown: &TokenBreakdown, total: usize, limit: usize) -> String {
    let mut text = String::new();
    for item in breakdown.items().into_iter().take(limit) {
        let pct = item
            .tokens
            .saturating_mul(100)
            .checked_div(total)
            .unwrap_or_default();
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

enum ModelPickerChoice {
    Profile(ModelProfile),
    ConfigureNew,
}

pub async fn select_model(
    sources: &ConfigSources,
    models: &mut ModelRegistry,
) -> anyhow::Result<ModelProfile> {
    let profiles = models.list().cloned().collect::<Vec<_>>();
    if profiles.is_empty() {
        return configure_and_insert_model(sources, models);
    }

    let choice = if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        let _raw = RawModeGuard::enter()?;
        select_model_with_arrows(&profiles)?
            .ok_or_else(|| anyhow::anyhow!("model selection canceled"))?
    } else {
        select_model_by_number(&profiles)?
    };

    match choice {
        ModelPickerChoice::Profile(profile) => Ok(profile),
        ModelPickerChoice::ConfigureNew => configure_and_insert_model(sources, models),
    }
}

fn select_model_by_number(profiles: &[ModelProfile]) -> anyhow::Result<ModelPickerChoice> {
    println!("configured models:");
    for (idx, profile) in profiles.iter().enumerate() {
        println!(
            "{}. {} ({}) @ {}",
            idx + 1,
            profile.name,
            profile.model,
            profile.base_url
        );
        println!();
    }
    println!("{}. configure new model", profiles.len() + 1);

    print!("select model [1]: ");
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let selected = input.trim().parse::<usize>().unwrap_or(1);
    if selected == profiles.len() + 1 {
        return Ok(ModelPickerChoice::ConfigureNew);
    }
    let index = selected.saturating_sub(1);
    profiles
        .get(index)
        .cloned()
        .map(ModelPickerChoice::Profile)
        .ok_or_else(|| anyhow::anyhow!("invalid model selection: {selected}"))
}

fn select_model_with_arrows(
    profiles: &[ModelProfile],
) -> anyhow::Result<Option<ModelPickerChoice>> {
    println!("\r\n{}", "models".with(VY_VIOLET).bold());
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
                if selected == profiles.len() {
                    return Ok(Some(ModelPickerChoice::ConfigureNew));
                }
                return Ok(profiles
                    .get(selected)
                    .cloned()
                    .map(ModelPickerChoice::Profile));
            }
            KeyCode::Up => {
                selected = if selected == 0 {
                    profiles.len()
                } else {
                    selected - 1
                };
                render_model_picker(profiles, selected, true)?;
            }
            KeyCode::Down => {
                selected = (selected + 1) % profiles.len().saturating_add(1);
                render_model_picker(profiles, selected, true)?;
            }
            KeyCode::Home => {
                selected = 0;
                render_model_picker(profiles, selected, true)?;
            }
            KeyCode::End => {
                selected = profiles.len();
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
    let row_count = profiles.len().saturating_mul(2).saturating_add(3);
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
                SetForegroundColor(VY_VIOLET),
                Print("> "),
                Print(row),
                ResetColor,
                Print("\r\n\r\n")
            )?;
        } else {
            execute!(
                std::io::stdout(),
                SetForegroundColor(VY_TEXT_MUTED),
                Print("  "),
                Print(row),
                ResetColor,
                Print("\r\n\r\n")
            )?;
        }
    }

    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine)
    )?;
    if selected == profiles.len() {
        execute!(
            std::io::stdout(),
            SetForegroundColor(VY_VIOLET),
            Print("> configure new model"),
            ResetColor,
            Print("\r\n\r\n")
        )?;
    } else {
        execute!(
            std::io::stdout(),
            SetForegroundColor(VY_TEXT_MUTED),
            Print("  configure new model"),
            ResetColor,
            Print("\r\n\r\n")
        )?;
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

fn configure_and_insert_model(
    sources: &ConfigSources,
    models: &mut ModelRegistry,
) -> anyhow::Result<ModelProfile> {
    let profile = configure_model_profile(sources)?;
    models.insert(profile.clone());
    Ok(profile)
}

fn configure_model_profile(sources: &ConfigSources) -> anyhow::Result<ModelProfile> {
    println!();
    println!("{}", "configure model".with(VY_VIOLET).bold());
    println!("profiles are saved to {}", sources.global_models.display());

    let name = prompt_with_default("profile name", "llama3")?;
    let base_url = prompt_with_default("base URL", "http://localhost:11434/v1")?;
    let model = prompt_with_default("model", "llama3.2")?;
    let api_key = prompt_optional("API key (optional)")?;

    let profile = ModelProfile {
        name,
        base_url,
        model,
        api_key,
    };
    crate::config::save_global_model_profile(sources, &profile)?;
    println!("saved {}", sources.global_models.display());
    Ok(profile)
}

fn prompt_with_default(label: &str, default: &str) -> anyhow::Result<String> {
    loop {
        print!("{label} [{default}]: ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let value = input.trim();
        if value.is_empty() {
            return Ok(default.to_string());
        }
        if !value.chars().any(char::is_whitespace) || label != "profile name" {
            return Ok(value.to_string());
        }
        println!("profile name cannot contain whitespace");
    }
}

fn prompt_optional(label: &str) -> anyhow::Result<String> {
    print!("{label}: ");
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

#[derive(Debug, Clone)]
pub(super) enum TuiUpdate {
    SummaryStart,
    SummaryDone {
        summary: Option<String>,
        retained_tokens: Option<TokenCount>,
    },
    AssistantStart,
    AssistantDelta(String),
    AssistantDone,
    AssistantInterrupted,
    ToolStarted {
        name: String,
        input: String,
    },
    ToolInputStart,
    ToolOk {
        name: String,
        preview: String,
    },
    ToolError {
        name: String,
        error: String,
    },
    ScratchpadStart,
    ScratchpadDone {
        summary: Option<String>,
        output_tokens: Option<TokenCount>,
    },
    Steering(String),
    Stats(String),
    Summary(String),
    Clarification(AskUserRequest),
}

enum ClarificationChoice {
    Option(usize),
    Other,
}

pub(super) enum ActiveTurnInput {
    Steering(String),
    Cancel,
    Clarification(AskUserResponse),
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

struct MouseCaptureGuard;

impl MouseCaptureGuard {
    fn enter() -> anyhow::Result<Self> {
        execute!(std::io::stdout(), EnableMouseCapture)?;
        Ok(Self)
    }
}

impl Drop for MouseCaptureGuard {
    fn drop(&mut self) {
        let _ = execute!(std::io::stdout(), DisableMouseCapture);
    }
}

struct CookedModeGuard;

impl CookedModeGuard {
    fn enter() -> anyhow::Result<Self> {
        execute!(std::io::stdout(), DisableBracketedPaste)?;
        disable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for CookedModeGuard {
    fn drop(&mut self) {
        let _ = enable_raw_mode();
        let _ = execute!(std::io::stdout(), EnableBracketedPaste);
    }
}

fn spawn_active_turn_listener(
    stop: Arc<AtomicBool>,
    pause: Arc<AtomicBool>,
    buffer: Arc<Mutex<String>>,
    input_tx: tokio::sync::mpsc::UnboundedSender<ActiveTurnInput>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            if pause.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(50));
                continue;
            }
            match event::poll(Duration::from_millis(50)) {
                Ok(true) => match event::read() {
                    Ok(Event::Key(key)) => match key.code {
                        KeyCode::Esc | KeyCode::Char('c')
                            if key.code == KeyCode::Esc
                                || key.modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            let _ = input_tx.send(ActiveTurnInput::Cancel);
                            break;
                        }
                        KeyCode::Enter => {
                            let message = buffer
                                .lock()
                                .map(|mut buffer| {
                                    let message = buffer.trim().to_string();
                                    buffer.clear();
                                    message
                                })
                                .unwrap_or_default();
                            let _ = clear_active_turn_input();
                            if !message.is_empty() {
                                let _ = input_tx.send(ActiveTurnInput::Steering(message));
                            }
                        }
                        KeyCode::Backspace => {
                            if let Ok(mut buffer) = buffer.lock() {
                                buffer.pop();
                                let _ = render_active_turn_input(&buffer);
                            }
                        }
                        KeyCode::Char(ch)
                            if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT =>
                        {
                            if let Ok(mut buffer) = buffer.lock() {
                                buffer.push(ch);
                                let _ = render_active_turn_input(&buffer);
                            }
                        }
                        _ => {}
                    },
                    Ok(Event::Paste(text)) => {
                        if let Ok(mut buffer) = buffer.lock() {
                            buffer.push_str(&text);
                            let _ = render_active_turn_input(&buffer);
                        }
                    }
                    _ => {}
                },
                Ok(false) => {}
                Err(_) => break,
            }
        }
    })
}

fn render_active_turn_input(input: &str) -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(VY_VIOLET),
        Print("steer> "),
        SetForegroundColor(VY_TECH_STRONG),
        Print(truncate_display(input, terminal_width().saturating_sub(8))),
        ResetColor
    )?;
    std::io::stdout().flush()?;
    Ok(())
}

fn clear_active_turn_input() -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine)
    )?;
    std::io::stdout().flush()?;
    Ok(())
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
            KeyCode::Char(ch)
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT =>
            {
                input.push(ch);
                render_clarification_freeform(
                    question,
                    question_index,
                    question_count,
                    &input,
                    true,
                )?;
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
    fn start(label: impl Into<String>, active_input: Arc<Mutex<String>>) -> Self {
        let label = label.into();
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let frames = ["·", "•", "●", "•"];
            let mut idx = 0;
            let started = Instant::now();
            while !thread_stop.load(Ordering::Relaxed) {
                let elapsed = started.elapsed().as_secs().max(1);
                let input = active_input
                    .lock()
                    .map(|input| input.clone())
                    .unwrap_or_default();
                if input.is_empty() {
                    let _ = execute!(
                        std::io::stdout(),
                        MoveToColumn(0),
                        Clear(ClearType::CurrentLine),
                        SetForegroundColor(VY_VIOLET),
                        Print(frames[idx % frames.len()]),
                        SetForegroundColor(VY_TEXT_MUTED),
                        Print(format!(
                            " working ({}s · type + enter to steer · esc to interrupt) — {}",
                            elapsed, label
                        )),
                        ResetColor
                    );
                } else {
                    let _ = render_active_turn_input(&input);
                }
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
        (String::from("  provider "), VY_TEXT_MUTED),
        (
            crate::tui::render::format_number(ledger.session_provider_tokens as isize),
            VY_TECH,
        ),
        (String::from("  fallback est. "), VY_TEXT_MUTED),
        (
            crate::tui::render::format_number(ledger.session_estimated_tokens as isize),
            if ledger.session_estimated_tokens > 0 {
                VY_TEXT_MUTED
            } else {
                VY_TEXT_DIM
            },
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
        (String::from("  memory overhead "), VY_TEXT_MUTED),
        (
            crate::tui::render::format_number(ledger.memory_overhead_tokens() as isize),
            VY_TEXT_DIM,
        ),
        (String::from("  net vs raw "), VY_TEXT_MUTED),
        (
            crate::tui::render::format_number(ledger.net_history_savings()),
            if ledger.net_history_savings() >= 0 {
                VY_SUCCESS
            } else {
                VY_RED
            },
        ),
        (String::from("  context "), VY_TEXT_MUTED),
        (
            format!(
                "{}/{} ({} available)",
                crate::tui::render::format_number(current_context as isize),
                crate::tui::render::format_number(max_context as isize),
                crate::tui::render::format_number(
                    max_context.saturating_sub(current_context) as isize
                )
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
            "provider counts win; fallback, context, would-be, and saved values are estimates",
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
    let mut mouse_capture = None;
    let (width, height) = size().unwrap_or((80, 24));
    let (_, origin_row) = crossterm::cursor::position()?;
    let mut layout = ComposerLayout::new(origin_row, width, height);
    render_composer_state(&state, status, &mut layout, (width, height))?;

    loop {
        let action = match event::read()? {
            Event::Key(key) => Some(handle_composer_key(key, &mut state, history)?),
            Event::Mouse(mouse) => {
                if !matches!(
                    mouse.kind,
                    MouseEventKind::Down(_)
                        | MouseEventKind::Up(_)
                        | MouseEventKind::ScrollUp
                        | MouseEventKind::ScrollDown
                ) {
                    continue;
                }
                let mut mouse_layout = layout;
                if let Ok((_, cursor_row)) = crossterm::cursor::position() {
                    mouse_layout.origin_row =
                        cursor_row.min(mouse_layout.terminal_height.saturating_sub(1));
                }
                handle_composer_mouse(mouse, &mut state, &mouse_layout)
            }
            Event::Paste(text) => {
                reset_history_navigation(&mut state);
                state.input.push_str(&text);
                reset_completion(&mut state.completion);
                Some(ComposerAction::Continue)
            }
            Event::Resize(width, height) => {
                render_composer_state(&state, status, &mut layout, (width, height))?;
                None
            }
            _ => None,
        };

        if let Some(action) = action {
            match action {
                ComposerAction::Continue => {
                    sync_mouse_capture(
                        &mut mouse_capture,
                        !slash_completion_matches(&state.input).is_empty(),
                    )?;
                    let terminal_size =
                        size().unwrap_or((layout.terminal_width, layout.terminal_height));
                    render_composer_state(&state, status, &mut layout, terminal_size)?;
                }
                ComposerAction::Submit => {
                    clear_composer(&layout)?;
                    print_user_block(&state.input, state.images.len())?;
                    return Ok(UserTurnInput {
                        text: state.input,
                        images: state.images,
                    });
                }
                ComposerAction::Exit => {
                    clear_composer(&layout)?;
                    return Ok(UserTurnInput {
                        text: "/exit".to_string(),
                        images: Vec::new(),
                    });
                }
            }
        }
    }
}

fn sync_mouse_capture(
    capture: &mut Option<MouseCaptureGuard>,
    palette_visible: bool,
) -> anyhow::Result<()> {
    match (palette_visible, capture.is_some()) {
        (true, false) => *capture = Some(MouseCaptureGuard::enter()?),
        (false, true) => *capture = None,
        _ => {}
    }
    Ok(())
}

fn render_composer_state(
    state: &ComposerState,
    status: &str,
    layout: &mut ComposerLayout,
    terminal_size: (u16, u16),
) -> anyhow::Result<()> {
    let completion_suffix = slash_completion_suffix(&state.input, &state.completion);
    let palette = slash_completion_matches(&state.input);
    let selected = active_slash_completion(&state.input, &state.completion);
    let selected_index = palette
        .iter()
        .position(|command| Some(command.name) == selected)
        .unwrap_or_default();
    *layout = ComposerLayout::for_render(
        layout.origin_row,
        terminal_size,
        palette.len(),
        selected_index,
    );
    render_composer(
        &state.input,
        state.images.len(),
        completion_suffix.as_deref(),
        &palette,
        selected,
        status,
        layout,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ComposerLayout {
    origin_row: u16,
    render_origin_row: u16,
    terminal_width: u16,
    terminal_height: u16,
    palette_start: usize,
    palette_len: usize,
    show_status: bool,
    status_spacer: bool,
}

impl ComposerLayout {
    fn new(origin_row: u16, width: u16, height: u16) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let origin_row = origin_row.min(height.saturating_sub(1));
        Self {
            origin_row,
            render_origin_row: origin_row,
            terminal_width: width,
            terminal_height: height,
            palette_start: 0,
            palette_len: 0,
            show_status: height >= 2,
            status_spacer: height >= 3,
        }
    }

    fn for_render(
        origin_row: u16,
        (width, height): (u16, u16),
        palette_count: usize,
        selected_index: usize,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let show_status = height >= 2;
        let status_spacer = height >= 3;
        let palette_capacity = usize::from(height.saturating_sub(3));
        let palette_len = palette_count.min(palette_capacity);
        let palette_start = palette_window_start(palette_count, palette_len, selected_index);
        let content_height =
            1 + palette_len + usize::from(show_status) + usize::from(status_spacer);
        let render_origin = origin_row.min(height.saturating_sub(1));
        let overflow =
            (usize::from(render_origin) + content_height).saturating_sub(usize::from(height));

        Self {
            origin_row: render_origin.saturating_sub(u16::try_from(overflow).unwrap_or(u16::MAX)),
            render_origin_row: render_origin,
            terminal_width: width,
            terminal_height: height,
            palette_start,
            palette_len,
            show_status,
            status_spacer,
        }
    }

    fn palette_index_at(self, column: u16, row: u16) -> Option<usize> {
        if column >= self.terminal_width {
            return None;
        }
        let offset = row.checked_sub(self.origin_row.saturating_add(1))? as usize;
        (offset < self.palette_len).then_some(self.palette_start + offset)
    }
}

fn palette_window_start(total: usize, visible: usize, selected: usize) -> usize {
    if visible == 0 || total <= visible {
        return 0;
    }
    selected
        .min(total.saturating_sub(1))
        .saturating_sub(visible / 2)
        .min(total - visible)
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
        KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            if state.input.trim().is_empty() {
                reset_history_navigation(state);
                state.input = "/".to_string();
                reset_completion(&mut state.completion);
            }
            Ok(ComposerAction::Continue)
        }
        KeyCode::F(1) => {
            if state.input.trim().is_empty() {
                reset_history_navigation(state);
                state.input = "/".to_string();
                reset_completion(&mut state.completion);
            }
            Ok(ComposerAction::Continue)
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
            if state.input.starts_with('/') {
                cycle_slash_completion(state, -1);
            } else {
                history_previous(state, history);
            }
            Ok(ComposerAction::Continue)
        }
        KeyCode::Down if key.modifiers.is_empty() => {
            if state.input.starts_with('/') {
                cycle_slash_completion(state, 1);
            } else {
                history_next(state, history);
            }
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

fn handle_composer_mouse(
    mouse: MouseEvent,
    state: &mut ComposerState,
    layout: &ComposerLayout,
) -> Option<ComposerAction> {
    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left) | MouseEventKind::Up(MouseButton::Left) => {
            let command_index = layout.palette_index_at(mouse.column, mouse.row)?;
            let command = slash_completion_matches(&state.input)
                .get(command_index)?
                .name;
            reset_history_navigation(state);
            state.input = command.to_string();
            reset_completion(&mut state.completion);
            Some(ComposerAction::Submit)
        }
        MouseEventKind::ScrollUp if layout.palette_index_at(mouse.column, mouse.row).is_some() => {
            cycle_slash_completion(state, -1);
            Some(ComposerAction::Continue)
        }
        MouseEventKind::ScrollDown
            if layout.palette_index_at(mouse.column, mouse.row).is_some() =>
        {
            cycle_slash_completion(state, 1);
            Some(ComposerAction::Continue)
        }
        _ => None,
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
    if let Some(command) = SLASH_COMMANDS.iter().find(|command| command.name == input) {
        return Some(command.name);
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
    matches
        .get(index % matches.len())
        .map(|command| command.name)
}

fn slash_completion_matches(input: &str) -> Vec<SlashCommand> {
    if !input.starts_with('/') || input.contains(' ') {
        return Vec::new();
    }
    SLASH_COMMANDS
        .iter()
        .copied()
        .filter(|command| command.name.starts_with(input))
        .collect::<Vec<_>>()
}

fn cycle_slash_completion(state: &mut ComposerState, direction: isize) {
    let matches = slash_completion_matches(&state.input);
    if matches.len() <= 1 {
        return;
    }
    if state.completion.prefix != state.input {
        state.completion.prefix = state.input.clone();
        state.completion.index = 0;
    }
    let len = matches.len() as isize;
    state.completion.index = (state.completion.index as isize + direction).rem_euclid(len) as usize;
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
    palette: &[SlashCommand],
    selected_command: Option<&str>,
    status: &str,
    layout: &ComposerLayout,
) -> anyhow::Result<()> {
    let mut stdout = std::io::stdout();
    let input_background = GRAPHITE_SURFACE_RAISED;
    let printable_width = usize::from(layout.terminal_width.saturating_sub(1));
    let prompt = truncate_display("> ", printable_width);
    let max_input_width = printable_width.saturating_sub(prompt.chars().count());
    let visible_input = truncate_display_start(&single_line_display(input), max_input_width);
    let cursor_column = u16::try_from(prompt.chars().count() + visible_input.chars().count())
        .unwrap_or(u16::MAX)
        .min(layout.terminal_width.saturating_sub(1));
    execute!(
        stdout,
        MoveTo(0, layout.render_origin_row),
        Clear(ClearType::FromCursorDown),
        SetBackgroundColor(input_background),
        Print(" ".repeat(printable_width)),
        MoveToColumn(0),
        SetBackgroundColor(input_background),
        SetForegroundColor(STEEL_BLUE),
        Print(&prompt),
        SetBackgroundColor(input_background),
        SetForegroundColor(VY_TECH_STRONG),
        Print(&visible_input)
    )?;
    let mut remaining_width = max_input_width.saturating_sub(visible_input.chars().count());
    if image_count > 0 {
        let attachment = truncate_display(
            &format!(
                "  [{image_count} image{}]",
                if image_count == 1 { "" } else { "s" }
            ),
            remaining_width,
        );
        execute!(
            stdout,
            SetBackgroundColor(input_background),
            SetForegroundColor(STEEL_BLUE),
            Print(&attachment)
        )?;
        remaining_width = remaining_width.saturating_sub(attachment.chars().count());
    }
    if let Some(completion_suffix) = completion_suffix {
        let completion_suffix = truncate_display(completion_suffix, remaining_width);
        execute!(
            stdout,
            SetBackgroundColor(input_background),
            SetForegroundColor(VY_TEXT_DIM),
            Print(completion_suffix)
        )?;
    }
    let palette_end = layout
        .palette_start
        .saturating_add(layout.palette_len)
        .min(palette.len());
    for command in &palette[layout.palette_start.min(palette_end)..palette_end] {
        let selected = selected_command == Some(command.name);
        let label = format!("{}{:<12}", if selected { "> " } else { "  " }, command.name);
        let visible_label = truncate_display(&label, printable_width);
        let description_width = printable_width.saturating_sub(visible_label.chars().count());
        let visible_description = truncate_display(command.description, description_width);
        execute!(
            stdout,
            ResetColor,
            Print("\r\n"),
            MoveToColumn(0),
            Clear(ClearType::CurrentLine),
            SetForegroundColor(if selected { VY_VIOLET } else { VY_TEXT_MUTED }),
            Print(visible_label),
            SetForegroundColor(VY_TEXT_DIM),
            Print(visible_description)
        )?;
    }
    if layout.show_status {
        execute!(stdout, ResetColor, Print("\r\n"))?;
        if layout.status_spacer {
            execute!(stdout, Print("\r\n"))?;
        }
        let status = if layout.palette_len < palette.len() {
            if layout.palette_len == 0 {
                format!(
                    "commands hidden ({}) · resize taller · {status}",
                    palette.len()
                )
            } else {
                format!(
                    "commands {}–{}/{} · click run · wheel/↑↓ select · {status}",
                    layout.palette_start + 1,
                    palette_end,
                    palette.len()
                )
            }
        } else if !palette.is_empty() {
            format!("click run · wheel/↑↓ select · {status}")
        } else {
            status.to_string()
        };
        execute!(
            stdout,
            MoveToColumn(0),
            Clear(ClearType::CurrentLine),
            SetForegroundColor(VY_TEXT_DIM),
            Print(truncate_display(&status, printable_width))
        )?;
    }
    execute!(stdout, ResetColor, MoveTo(cursor_column, layout.origin_row))?;
    stdout.flush()?;
    Ok(())
}

fn clear_composer(layout: &ComposerLayout) -> anyhow::Result<()> {
    execute!(
        std::io::stdout(),
        MoveTo(0, layout.origin_row),
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
    let width = terminal_width().clamp(56, 78);
    let border = "-".repeat(width.saturating_sub(2));
    print_welcome_line(format!("+{border}+").with(STEEL_BLUE))?;
    print_welcome_line(banner_line(" __     __ __   __ ____  _   _ ", width).with(VY_VIOLET))?;
    print_welcome_line(
        banner_line(" \\ \\   / / \\ \\ / /|  _ \\| \\ | |", width).with(VY_VIOLET),
    )?;
    print_welcome_line(banner_line("  \\ \\ / /   \\ V / | |_) |  \\| |", width).with(VY_VIOLET))?;
    print_welcome_line(banner_line("   \\ V /     | |  |  _ <| |\\  |", width).with(VY_VIOLET))?;
    print_welcome_line(banner_line("    \\_/      |_|  |_| \\_\\_| \\_|", width).with(VY_VIOLET))?;
    print_welcome_line(format!("+{border}+").with(STEEL_BLUE))?;
    print_welcome_line(format!(
        "{} {}  {}",
        "model".with(VY_TEXT_DIM),
        app.model.name.as_str().with(STEEL_BLUE),
        format!("context {}", app.config.context.max_tokens).with(VY_TEXT_DIM)
    ))?;
    print_welcome_line("type / or press Ctrl+O for commands".with(VY_TEXT_DIM))?;
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

fn print_steering_block(text: &str) -> anyhow::Result<()> {
    for line in text.lines() {
        print_block_line(
            "steer",
            line,
            GRAPHITE_SURFACE_RAISED,
            VY_VIOLET,
            VY_TECH_STRONG,
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

fn exchange_user_input_with_steering(
    text: &str,
    image_count: usize,
    steering_inputs: &[String],
) -> String {
    let mut combined = exchange_user_input(text, image_count);
    for steering in steering_inputs {
        combined.push_str("\nlive steering: ");
        combined.push_str(steering);
    }
    combined
}

fn first_non_empty_line(value: &str) -> Option<&str> {
    value.lines().find(|line| !line.trim().is_empty())
}

fn truncate_display(value: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
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

fn truncate_display_start(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        return value.to_string();
    }
    if max_chars == 0 {
        return String::new();
    }
    let mut out = String::from("…");
    out.extend(value.chars().skip(char_count - max_chars.saturating_sub(1)));
    out
}

fn single_line_display(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect()
}

fn select_model_inline(
    sources: &ConfigSources,
    models: &mut ModelRegistry,
) -> anyhow::Result<Option<ModelProfile>> {
    let profiles = models.list().cloned().collect::<Vec<_>>();
    if profiles.is_empty() {
        let _cooked = CookedModeGuard::enter()?;
        return configure_and_insert_model(sources, models).map(Some);
    }

    match select_model_with_arrows(&profiles)? {
        Some(ModelPickerChoice::Profile(profile)) => Ok(Some(profile)),
        Some(ModelPickerChoice::ConfigureNew) => {
            let _cooked = CookedModeGuard::enter()?;
            configure_and_insert_model(sources, models).map(Some)
        }
        None => Ok(None),
    }
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
    fn slash_prefix_lists_the_full_command_palette() {
        let commands = slash_completion_matches("/");
        let names = commands
            .iter()
            .map(|command| command.name)
            .collect::<Vec<_>>();

        assert!(names.contains(&"/help"));
        assert!(names.contains(&"/stats"));
        assert!(names.contains(&"/context"));
        assert!(names.contains(&"/summary"));
        assert!(names.contains(&"/scratchpad"));
        assert!(names.contains(&"/debug"));
    }

    #[test]
    fn control_o_opens_command_palette_from_empty_composer() {
        let mut state = ComposerState::default();

        let action = handle_composer_key(
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &mut state,
            &[],
        )
        .unwrap();

        assert!(matches!(action, ComposerAction::Continue));
        assert_eq!(state.input, "/");
        assert!(!slash_completion_matches(&state.input).is_empty());
    }

    #[test]
    fn command_palette_uses_arrows_to_change_selection() {
        let mut state = ComposerState {
            input: "/".to_string(),
            ..Default::default()
        };
        let first = active_slash_completion(&state.input, &state.completion).unwrap();

        handle_composer_key(
            KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
            &mut state,
            &[],
        )
        .unwrap();

        assert_ne!(
            active_slash_completion(&state.input, &state.completion),
            Some(first)
        );
    }

    #[test]
    fn command_palette_layout_tracks_narrow_and_resized_terminals() {
        let full = ComposerLayout::for_render(10, (120, 40), SLASH_COMMANDS.len(), 0);
        assert_eq!(full.palette_start, 0);
        assert_eq!(full.palette_len, SLASH_COMMANDS.len());
        assert_eq!(full.origin_row, 10);
        assert_eq!(full.palette_index_at(119, 11), Some(0));

        let resized = ComposerLayout::for_render(
            full.origin_row,
            (24, 7),
            SLASH_COMMANDS.len(),
            SLASH_COMMANDS.len() - 1,
        );
        assert_eq!(resized.palette_len, 4);
        assert_eq!(resized.palette_start, SLASH_COMMANDS.len() - 4);
        assert_eq!(resized.origin_row, 0);
        assert_eq!(
            resized.palette_index_at(23, 1),
            Some(SLASH_COMMANDS.len() - 4)
        );
        assert_eq!(
            resized.palette_index_at(23, 4),
            Some(SLASH_COMMANDS.len() - 1)
        );
        assert_eq!(resized.palette_index_at(24, 4), None);
        assert_eq!(resized.palette_index_at(2, 5), None);

        let one_row = ComposerLayout::for_render(4, (1, 1), SLASH_COMMANDS.len(), 0);
        assert_eq!(one_row.origin_row, 0);
        assert_eq!(one_row.palette_len, 0);
        assert!(!one_row.show_status);
    }

    #[test]
    fn pressing_or_releasing_a_visible_palette_row_submits_that_command() {
        let layout = ComposerLayout::for_render(5, (80, 24), SLASH_COMMANDS.len(), 0);
        for kind in [
            MouseEventKind::Down(MouseButton::Left),
            MouseEventKind::Up(MouseButton::Left),
        ] {
            let mut state = ComposerState {
                input: "/".to_string(),
                ..Default::default()
            };
            let click = MouseEvent {
                kind,
                column: 40,
                row: layout.origin_row + 3,
                modifiers: KeyModifiers::NONE,
            };

            let action = handle_composer_mouse(click, &mut state, &layout);

            assert!(matches!(action, Some(ComposerAction::Submit)));
            assert_eq!(state.input, "/context");
        }
    }

    #[test]
    fn mouse_wheel_only_changes_selection_over_the_palette() {
        let mut state = ComposerState {
            input: "/".to_string(),
            ..Default::default()
        };
        let layout = ComposerLayout::for_render(2, (80, 24), SLASH_COMMANDS.len(), 0);
        let over_palette = MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 5,
            row: layout.origin_row + 1,
            modifiers: KeyModifiers::NONE,
        };
        let outside_palette = MouseEvent {
            row: layout.origin_row,
            ..over_palette
        };

        assert!(matches!(
            handle_composer_mouse(over_palette, &mut state, &layout),
            Some(ComposerAction::Continue)
        ));
        assert_eq!(
            active_slash_completion(&state.input, &state.completion),
            Some("/stats")
        );
        assert!(handle_composer_mouse(outside_palette, &mut state, &layout).is_none());
        assert_eq!(
            active_slash_completion(&state.input, &state.completion),
            Some("/stats")
        );
    }

    #[test]
    fn composer_text_is_single_line_and_width_bounded() {
        assert_eq!(
            single_line_display("first\nsecond\tline"),
            "first second line"
        );
        assert_eq!(truncate_display("abc", 0), "");
        assert_eq!(truncate_display_start("abcdefgh", 5), "…efgh");
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
