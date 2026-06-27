use crate::agent::context::ContextManager;
use crate::agent::prompt::build_agent_prompt;
use crate::agent::tokens::{
    TokenBreakdown, TokenLedger, TurnUsage, estimate_assistant_output_tokens,
    estimate_chat_request_breakdown, estimate_messages_breakdown, estimate_text_tokens,
    estimate_unpruned_request_tokens,
};
use crate::agent::transcript::{Exchange, truncate};
use crate::agent::turn::{
    TurnScratchpad, build_fitted_turn_scratchpad_update_messages, build_turn_messages,
    prepare_next_turn_context, turn_scratchpad_update_source,
};
use crate::cli::EvalArgs;
use crate::config::{ConfigSources, EffectiveConfig, ModelProfile, ModelRegistry};
use crate::llm::{
    ChatCompletionRequest, ChatMessage, ImageAttachment, LlmError, OpenAiClient, StreamEvent,
    ToolCall,
};
use crate::mcp::McpRegistry;
use crate::skills::SkillRegistry;
use crate::tools::{MachineManifest, ToolRegistry, ToolResult};
use crate::vision;
use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::process::Command;
use tokio::time::timeout;

const DEFAULT_MAX_TOOL_ROUNDS: usize = 64;
const ASSERTION_COMMAND_TIMEOUT_SECONDS: u64 = 120;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalSuite {
    pub name: String,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub cases: Vec<EvalCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvalCase {
    pub id: String,
    #[serde(default)]
    pub description: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_turns: Option<usize>,
    #[serde(default)]
    pub assertions: Vec<EvalAssertion>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvalAssertion {
    AssistantContains {
        value: String,
    },
    AssistantNotContains {
        value: String,
    },
    ToolCalled {
        name: String,
    },
    ToolCalledAtLeast {
        name: String,
        count: usize,
    },
    ToolNotCalled {
        name: String,
    },
    FileExists {
        path: PathBuf,
    },
    FileContains {
        path: PathBuf,
        value: String,
    },
    CommandSucceeds {
        command: String,
    },
    Judge {
        prompt: String,
        model: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize)]
struct EvalRunSummary {
    suite: String,
    suite_path: String,
    output_dir: String,
    total: usize,
    passed: usize,
    failed: usize,
    duration_ms: u128,
    cases: Vec<EvalCaseSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct EvalCaseSummary {
    id: String,
    model: String,
    passed: bool,
    duration_ms: u128,
    sent_tokens: usize,
    would_be_tokens: usize,
    saved_tokens: isize,
    trace_dir: String,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EvalTrace {
    case: EvalCase,
    model: ModelProfile,
    passed: bool,
    duration_ms: u128,
    final_assistant: String,
    requests: Vec<EvalRequestTrace>,
    events: Vec<EvalTraceEvent>,
    tool_calls: Vec<ToolCall>,
    tool_results: Vec<ToolResult>,
    assertions: Vec<AssertionOutcome>,
    stats: TokenLedger,
    debug: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EvalRequestTrace {
    label: String,
    messages: Vec<ChatMessage>,
    tool_count: usize,
    estimated_input_tokens: usize,
    response_text: String,
    tool_calls: Vec<ToolCall>,
    output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum EvalTraceEvent {
    AssistantDelta {
        round: usize,
        text: String,
    },
    ToolStarted {
        round: usize,
        name: String,
    },
    ToolOk {
        round: usize,
        result: ToolResult,
    },
    ToolError {
        round: usize,
        name: String,
        error: String,
    },
    Scratchpad {
        round: usize,
        summary: String,
    },
}

#[derive(Debug, Clone, Serialize)]
struct AssertionOutcome {
    assertion: EvalAssertion,
    passed: bool,
    message: String,
}

pub async fn run(args: EvalArgs, context_override: Option<usize>) -> anyhow::Result<i32> {
    let start = Instant::now();
    let suite_path = args.suite.clone();
    let suite = load_suite(&suite_path)?;
    validate_suite(&suite)?;
    let selected_cases = selected_cases(&suite, args.case.as_deref())?;

    if args.dry_run {
        println!(
            "eval suite '{}' is valid ({} case{})",
            suite.name,
            selected_cases.len(),
            if selected_cases.len() == 1 { "" } else { "s" }
        );
        return Ok(0);
    }

    eprintln!(
        "warning: eval cases run in the current repository and may modify files; traces will be written under .vyrn/eval-runs unless --output is set"
    );
    eprintln!(
        "warning: token budget checks use local estimates; first requests, image payloads, provider tokenizer differences, and future MCP tool schemas may still need separate hardening"
    );

    let cwd = std::env::current_dir()?;
    let sources = ConfigSources::discover(cwd.clone())?;
    let mut config = EffectiveConfig::load(&sources)?;
    if let Some(max_tokens) = context_override {
        config.context.max_tokens = max_tokens;
    }
    let models = crate::config::load_model_profiles(&sources)?;
    let output_dir = args.output.unwrap_or_else(default_output_dir);
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let mut summaries = Vec::with_capacity(selected_cases.len());
    let mut passed = 0;
    let mut failed = 0;

    for case in selected_cases {
        let case_start = Instant::now();
        let trace_dir = output_dir.join(&case.id);
        std::fs::create_dir_all(&trace_dir)
            .with_context(|| format!("failed to create {}", trace_dir.display()))?;
        let model = resolve_case_model(&models, &suite, case, args.model.as_deref())?;
        let mut runner = EvalAgentRunner::new(
            sources.clone(),
            config.clone(),
            models.clone(),
            model.clone(),
            !args.no_debug,
        )?;
        let mut trace = runner.run_case(case.clone()).await;
        trace.duration_ms = case_start.elapsed().as_millis();
        if trace.error.is_none() {
            trace.assertions = evaluate_assertions(&trace, &models, &model).await;
            if trace.assertions.iter().any(|assertion| !assertion.passed) {
                trace.error = Some("one or more assertions failed".to_string());
            }
        }
        trace.passed = trace.error.is_none();
        write_case_trace(&trace_dir, &trace)?;

        if trace.passed {
            passed += 1;
        } else {
            failed += 1;
        }

        let sent_tokens = trace.stats.session_sent;
        let would_be_tokens = trace.stats.session_would_be;
        let saved_tokens = trace.stats.session_saved;
        let summary = EvalCaseSummary {
            id: case.id.clone(),
            model: model.name.clone(),
            passed: trace.passed,
            duration_ms: trace.duration_ms,
            sent_tokens,
            would_be_tokens,
            saved_tokens,
            trace_dir: trace_dir.display().to_string(),
            error: trace.error.clone(),
        };
        print_case_summary(&summary);
        summaries.push(summary);
    }

    let run_summary = EvalRunSummary {
        suite: suite.name,
        suite_path: suite_path.display().to_string(),
        output_dir: output_dir.display().to_string(),
        total: summaries.len(),
        passed,
        failed,
        duration_ms: start.elapsed().as_millis(),
        cases: summaries,
    };
    write_summary(&output_dir, &run_summary)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&run_summary)?);
    } else {
        println!(
            "eval result: {} passed, {} failed, traces: {}",
            run_summary.passed, run_summary.failed, run_summary.output_dir
        );
    }

    Ok(if run_summary.failed == 0 { 0 } else { 1 })
}

fn load_suite(path: &Path) -> anyhow::Result<EvalSuite> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn validate_suite(suite: &EvalSuite) -> anyhow::Result<()> {
    if suite.name.trim().is_empty() {
        anyhow::bail!("eval suite name cannot be empty");
    }
    if suite.cases.is_empty() {
        anyhow::bail!("eval suite must include at least one case");
    }
    let mut ids = BTreeSet::new();
    for case in &suite.cases {
        if case.id.trim().is_empty() {
            anyhow::bail!("eval case id cannot be empty");
        }
        if !ids.insert(case.id.as_str()) {
            anyhow::bail!("duplicate eval case id '{}'", case.id);
        }
        if case.prompt.trim().is_empty() {
            anyhow::bail!("eval case '{}' prompt cannot be empty", case.id);
        }
        if case.assertions.is_empty() {
            anyhow::bail!(
                "eval case '{}' must include at least one assertion",
                case.id
            );
        }
    }
    Ok(())
}

fn selected_cases<'a>(
    suite: &'a EvalSuite,
    case_id: Option<&str>,
) -> anyhow::Result<Vec<&'a EvalCase>> {
    if let Some(case_id) = case_id {
        let case = suite
            .cases
            .iter()
            .find(|case| case.id == case_id)
            .ok_or_else(|| anyhow::anyhow!("eval case '{case_id}' was not found"))?;
        return Ok(vec![case]);
    }
    Ok(suite.cases.iter().collect())
}

fn resolve_case_model(
    models: &ModelRegistry,
    suite: &EvalSuite,
    case: &EvalCase,
    override_model: Option<&str>,
) -> anyhow::Result<ModelProfile> {
    let selected = override_model
        .or(case.model.as_deref())
        .or(suite.default_model.as_deref());
    if let Some(selected) = selected {
        return models
            .get(selected)
            .ok_or_else(|| anyhow::anyhow!("model profile '{selected}' was not found"));
    }
    models
        .first()
        .ok_or_else(|| anyhow::anyhow!("no model profiles configured"))
}

fn default_output_dir() -> PathBuf {
    PathBuf::from(".vyrn")
        .join("eval-runs")
        .join(unix_timestamp_millis().to_string())
}

fn print_case_summary(summary: &EvalCaseSummary) {
    let status = if summary.passed { "PASS" } else { "FAIL" };
    println!(
        "{status} {} model={} duration={}ms sent={} trace={}",
        summary.id, summary.model, summary.duration_ms, summary.sent_tokens, summary.trace_dir
    );
    if let Some(error) = &summary.error {
        println!("  {error}");
    }
}

fn write_summary(output_dir: &Path, summary: &EvalRunSummary) -> anyhow::Result<()> {
    let path = output_dir.join("summary.json");
    let raw = serde_json::to_string_pretty(summary)?;
    std::fs::write(&path, raw).with_context(|| format!("failed to write {}", path.display()))
}

fn write_case_trace(trace_dir: &Path, trace: &EvalTrace) -> anyhow::Result<()> {
    let raw = serde_json::to_string_pretty(trace)?;
    std::fs::write(trace_dir.join("trace.json"), raw)
        .with_context(|| format!("failed to write {}", trace_dir.join("trace.json").display()))?;
    std::fs::write(trace_dir.join("transcript.md"), render_transcript(trace)).with_context(
        || {
            format!(
                "failed to write {}",
                trace_dir.join("transcript.md").display()
            )
        },
    )?;
    std::fs::write(trace_dir.join("debug.log"), trace.debug.join("\n"))
        .with_context(|| format!("failed to write {}", trace_dir.join("debug.log").display()))?;
    Ok(())
}

fn render_transcript(trace: &EvalTrace) -> String {
    let mut text = String::new();
    text.push_str(&format!("# {}\n\n", trace.case.id));
    if let Some(description) = &trace.case.description {
        text.push_str(description);
        text.push_str("\n\n");
    }
    text.push_str("## Prompt\n\n");
    text.push_str(&trace.case.prompt);
    text.push_str("\n\n## Events\n\n");
    for event in &trace.events {
        match event {
            EvalTraceEvent::AssistantDelta { round, text: delta } => {
                text.push_str(&format!("- round {round} assistant: {}\n", delta.trim()));
            }
            EvalTraceEvent::ToolStarted { round, name } => {
                text.push_str(&format!("- round {round} tool started: `{name}`\n"));
            }
            EvalTraceEvent::ToolOk { round, result } => {
                text.push_str(&format!(
                    "- round {round} tool ok: `{}`\n\n```text\n{}\n```\n",
                    result.name,
                    truncate(&result.content, 2000)
                ));
            }
            EvalTraceEvent::ToolError { round, name, error } => {
                text.push_str(&format!("- round {round} tool error: `{name}` {error}\n"));
            }
            EvalTraceEvent::Scratchpad { round, summary } => {
                text.push_str(&format!(
                    "- round {round} scratchpad:\n\n{}\n",
                    summary.trim()
                ));
            }
        }
    }
    text.push_str("\n## Final Assistant\n\n");
    text.push_str(&trace.final_assistant);
    text.push_str("\n\n## Assertions\n\n");
    for assertion in &trace.assertions {
        let status = if assertion.passed { "PASS" } else { "FAIL" };
        text.push_str(&format!("- {status}: {}\n", assertion.message));
    }
    if let Some(error) = &trace.error {
        text.push_str(&format!("\n## Error\n\n{error}\n"));
    }
    text
}

async fn evaluate_assertions(
    trace: &EvalTrace,
    models: &ModelRegistry,
    default_model: &ModelProfile,
) -> Vec<AssertionOutcome> {
    let mut outcomes = Vec::with_capacity(trace.case.assertions.len());
    for assertion in &trace.case.assertions {
        outcomes.push(evaluate_assertion(assertion, trace, models, default_model).await);
    }
    outcomes
}

async fn evaluate_assertion(
    assertion: &EvalAssertion,
    trace: &EvalTrace,
    models: &ModelRegistry,
    default_model: &ModelProfile,
) -> AssertionOutcome {
    match assertion {
        EvalAssertion::AssistantContains { value } => {
            let passed = trace.final_assistant.contains(value);
            AssertionOutcome {
                assertion: assertion.clone(),
                passed,
                message: format!("assistant contains '{value}'"),
            }
        }
        EvalAssertion::AssistantNotContains { value } => {
            let passed = !trace.final_assistant.contains(value);
            AssertionOutcome {
                assertion: assertion.clone(),
                passed,
                message: format!("assistant does not contain '{value}'"),
            }
        }
        EvalAssertion::ToolCalled { name } => {
            let passed = trace
                .tool_calls
                .iter()
                .any(|call| call.function.name == *name);
            AssertionOutcome {
                assertion: assertion.clone(),
                passed,
                message: format!("tool '{name}' was called"),
            }
        }
        EvalAssertion::ToolCalledAtLeast { name, count } => {
            let actual = trace
                .tool_calls
                .iter()
                .filter(|call| call.function.name == *name)
                .count();
            AssertionOutcome {
                assertion: assertion.clone(),
                passed: actual >= *count,
                message: format!(
                    "tool '{name}' was called {actual} time(s), expected at least {count}"
                ),
            }
        }
        EvalAssertion::ToolNotCalled { name } => {
            let passed = !trace
                .tool_calls
                .iter()
                .any(|call| call.function.name == *name);
            AssertionOutcome {
                assertion: assertion.clone(),
                passed,
                message: format!("tool '{name}' was not called"),
            }
        }
        EvalAssertion::FileExists { path } => {
            let passed = path.exists();
            AssertionOutcome {
                assertion: assertion.clone(),
                passed,
                message: format!("file '{}' exists", path.display()),
            }
        }
        EvalAssertion::FileContains { path, value } => {
            let passed = std::fs::read_to_string(path)
                .map(|content| content.contains(value))
                .unwrap_or(false);
            AssertionOutcome {
                assertion: assertion.clone(),
                passed,
                message: format!("file '{}' contains '{}'", path.display(), value),
            }
        }
        EvalAssertion::CommandSucceeds { command } => {
            let passed = command_succeeds(command).await.unwrap_or(false);
            AssertionOutcome {
                assertion: assertion.clone(),
                passed,
                message: format!("command succeeds: {command}"),
            }
        }
        EvalAssertion::Judge { prompt, model } => {
            let result =
                judge_assertion(prompt, model.as_deref(), trace, models, default_model).await;
            match result {
                Ok(passed) => AssertionOutcome {
                    assertion: assertion.clone(),
                    passed,
                    message: format!("judge assertion: {prompt}"),
                },
                Err(error) => AssertionOutcome {
                    assertion: assertion.clone(),
                    passed: false,
                    message: format!("judge assertion failed to run: {error}"),
                },
            }
        }
    }
}

async fn command_succeeds(command: &str) -> anyhow::Result<bool> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut child = Command::new(shell);
    child.arg("-lc").arg(command);
    child.kill_on_drop(true);
    let status = timeout(
        Duration::from_secs(ASSERTION_COMMAND_TIMEOUT_SECONDS),
        child.status(),
    )
    .await??;
    Ok(status.success())
}

async fn judge_assertion(
    prompt: &str,
    model: Option<&str>,
    trace: &EvalTrace,
    models: &ModelRegistry,
    default_model: &ModelProfile,
) -> anyhow::Result<bool> {
    let profile = if let Some(model) = model {
        models
            .get(model)
            .ok_or_else(|| anyhow::anyhow!("judge model profile '{model}' was not found"))?
    } else {
        default_model.clone()
    };
    let client = OpenAiClient::new(profile);
    let response = client
        .complete_chat(ChatCompletionRequest {
            model: String::new(),
            messages: vec![
                ChatMessage::system(
                    "You judge a terminal agent eval. Return PASS or FAIL on the first line, then a concise reason.",
                ),
                ChatMessage::user(format!(
                    "Assertion:\n{prompt}\n\nUser prompt:\n{}\n\nFinal assistant:\n{}\n\nTool calls:\n{}",
                    trace.case.prompt,
                    trace.final_assistant,
                    serde_json::to_string(&trace.tool_calls).unwrap_or_default()
                )),
            ],
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
        })
        .await?;
    let text = response
        .choices
        .first()
        .and_then(|choice| choice.message.content_text())
        .unwrap_or_default()
        .trim()
        .to_ascii_uppercase();
    Ok(text.starts_with("PASS") && !text.starts_with("FAIL"))
}

struct EvalAgentRunner {
    config: EffectiveConfig,
    client: OpenAiClient,
    tools: ToolRegistry,
    manifest: MachineManifest,
    skills: SkillRegistry,
    mcp: McpRegistry,
    context: ContextManager,
    stats: TokenLedger,
    debug_enabled: bool,
    debug: Vec<String>,
}

impl EvalAgentRunner {
    fn new(
        sources: ConfigSources,
        config: EffectiveConfig,
        _models: ModelRegistry,
        model: ModelProfile,
        debug_enabled: bool,
    ) -> anyhow::Result<Self> {
        let skills = SkillRegistry::discover(&sources)?;
        let mcp = McpRegistry::load(&sources)?;
        let manifest = MachineManifest::scan(&skills, &mcp);
        let context = ContextManager::new(
            config.context.max_tokens,
            config.context.summary_aggressiveness,
        );
        Ok(Self {
            client: OpenAiClient::new(model),
            config,
            tools: ToolRegistry::core(),
            manifest,
            skills,
            mcp,
            context,
            stats: TokenLedger::default(),
            debug_enabled,
            debug: Vec::new(),
        })
    }

    async fn run_case(&mut self, case: EvalCase) -> EvalTrace {
        let started = Instant::now();
        let mut trace = EvalTrace {
            case: case.clone(),
            model: self.client.profile().clone(),
            passed: false,
            duration_ms: 0,
            final_assistant: String::new(),
            requests: Vec::new(),
            events: Vec::new(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            assertions: Vec::new(),
            stats: TokenLedger::default(),
            debug: Vec::new(),
            error: None,
        };

        self.debug_log(format!(
            "eval_case_start id={} model={} base_url={} context={}",
            case.id,
            self.client.profile().name,
            self.client.profile().base_url,
            self.config.context.max_tokens
        ));

        let result = self.run_prompt(&case, &mut trace).await;
        if let Err(error) = result {
            trace.error = Some(error.to_string());
        }
        trace.duration_ms = started.elapsed().as_millis();
        trace.stats = self.stats.clone();
        trace.debug = self.debug.clone();
        trace
    }

    async fn run_prompt(&mut self, case: &EvalCase, trace: &mut EvalTrace) -> Result<(), LlmError> {
        let text_images = vision::attachments_from_text(&case.prompt)
            .await
            .map_err(|error| LlmError::Input(error.to_string()))?;
        let mut images = Vec::new();
        images.extend(text_images);
        dedupe_images(&mut images);
        images.truncate(vision::MAX_IMAGES_PER_MESSAGE);

        let initial_prompt = build_agent_prompt(
            &self.tools,
            &self.manifest,
            self.context.summary(),
            &case.prompt,
            &images,
        );
        self.debug_log(format!(
            "turn_start user_tokens={} images={} initial_prompt_tokens={} raw_history_tokens={} has_summary={}",
            estimate_text_tokens(&case.prompt),
            images.len(),
            initial_prompt.estimated_tokens.tokens,
            self.context.raw_history_tokens(),
            self.context.summary().is_some()
        ));
        let mut usage = TurnUsage::default();

        if let Some(summary_usage) = self
            .context
            .refresh_summary(&self.client, initial_prompt.estimated_tokens.tokens)
            .await?
        {
            let summary_total = summary_usage.input_tokens + summary_usage.output_tokens;
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

        let prompt = build_agent_prompt(
            &self.tools,
            &self.manifest,
            self.context.summary(),
            &case.prompt,
            &images,
        );
        usage.context_tokens = prompt.estimated_tokens.tokens;
        let base_messages = prompt.messages;
        let mut scratchpad = TurnScratchpad::default();
        let mut current_tool_batch = Vec::new();
        let mut last_request_messages =
            build_turn_messages(&base_messages, &scratchpad, &current_tool_batch);
        let mut assistant_text = String::new();
        let max_rounds = case.max_turns.unwrap_or(DEFAULT_MAX_TOOL_ROUNDS).max(1);
        let mut hit_tool_round_limit = false;

        for round in 0..max_rounds {
            let tool_schemas = self.tools.schemas();
            let messages = build_turn_messages(&base_messages, &scratchpad, &current_tool_batch);
            let request_breakdown = estimate_chat_request_breakdown(&messages, &tool_schemas);
            let request_tokens = request_breakdown.total();
            let request_would_be = estimate_unpruned_request_tokens(
                &request_breakdown,
                self.context.raw_history_tokens(),
            );
            self.debug_log(format!(
                "agent_request round={} request_tokens={} would_be={} max_context={} messages={} tool_schema_tokens={} summaries={} tool_outputs={} assistant_context={}",
                round,
                request_tokens,
                request_would_be,
                self.config.context.max_tokens,
                messages.len(),
                request_breakdown.tool_schemas,
                request_breakdown.summaries,
                request_breakdown.tool_call_outputs,
                request_breakdown.assistant_context
            ));
            let has_chained_context = round > 0
                || !scratchpad.summary.trim().is_empty()
                || !current_tool_batch.is_empty();
            if has_chained_context && request_tokens > self.config.context.max_tokens {
                return Err(LlmError::Input(format!(
                    "context budget exceeded before chained tool request: estimated {request_tokens} tokens exceeds configured {}",
                    self.config.context.max_tokens
                )));
            }

            let mut response_text = String::new();
            let response = self
                .client
                .stream_chat(
                    ChatCompletionRequest {
                        model: String::new(),
                        messages: messages.clone(),
                        tools: tool_schemas.clone(),
                        tool_choice: None,
                        stream: true,
                    },
                    |event| match event {
                        StreamEvent::TextDelta(delta) => {
                            response_text.push_str(&delta);
                            trace
                                .events
                                .push(EvalTraceEvent::AssistantDelta { round, text: delta });
                        }
                        StreamEvent::ToolCallDone(call) => {
                            trace.events.push(EvalTraceEvent::ToolStarted {
                                round,
                                name: call.function.name,
                            });
                        }
                        StreamEvent::Finished => {}
                    },
                )
                .await?;
            last_request_messages = messages.clone();

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
            trace.requests.push(EvalRequestTrace {
                label: format!("agent-{round}"),
                messages,
                tool_count: tool_schemas.len(),
                estimated_input_tokens: request_tokens,
                response_text,
                tool_calls: tool_calls.clone(),
                output_tokens,
            });
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
                let result = self.execute_tool_call(&call).await;
                let mut incremental_batch = vec![assistant_tool_message.clone()];
                match &result {
                    Ok(tool_result) => {
                        let tool_message = ChatMessage::tool(
                            call.id.clone(),
                            truncate(&tool_result.content, 8000),
                        );
                        incremental_batch.push(tool_message.clone());
                        current_tool_batch.push(tool_message);
                        trace.events.push(EvalTraceEvent::ToolOk {
                            round,
                            result: tool_result.clone(),
                        });
                        trace.tool_results.push(tool_result.clone());
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
                        let tool_message = ChatMessage::tool(call.id.clone(), content);
                        incremental_batch.push(tool_message.clone());
                        current_tool_batch.push(tool_message);
                        trace.events.push(EvalTraceEvent::ToolError {
                            round,
                            name: call.function.name.clone(),
                            error: error.to_string(),
                        });
                    }
                }
                trace.tool_calls.push(call);
                scratchpad = self
                    .update_turn_scratchpad(
                        &scratchpad,
                        &incremental_batch,
                        &assistant_tool_message,
                        &mut usage,
                    )
                    .await?;
                let next_context = prepare_next_turn_context(
                    &base_messages,
                    &scratchpad,
                    &current_tool_batch,
                    &tool_schemas,
                    self.config.context.max_tokens,
                )?;
                scratchpad = next_context.scratchpad;
                current_tool_batch = next_context.tool_batch;
                trace.events.push(EvalTraceEvent::Scratchpad {
                    round,
                    summary: scratchpad.summary.clone(),
                });
                let preparation = next_context.preparation;
                self.debug_log(format!(
                    "tool_chain_prepare round={} tool_index={} before_tokens={} after_tokens={} threshold={} max_context={} scratchpad_tokens={} current_batch_messages={}",
                    round,
                    tool_index,
                    preparation.before_tokens,
                    preparation.after_tokens,
                    preparation.threshold,
                    preparation.max_tokens,
                    estimate_text_tokens(&scratchpad.summary),
                    current_tool_batch.len()
                ));
            }
            if round + 1 == max_rounds {
                hit_tool_round_limit = true;
            }
        }

        self.context.set_previous_exchange(Exchange {
            user_input: exchange_user_input(&case.prompt, images.len()),
            assistant_text: assistant_text.clone(),
            tool_calls: trace.tool_calls.clone(),
            tool_results: trace.tool_results.clone(),
        });
        usage.context_tokens =
            estimate_chat_request_breakdown(&last_request_messages, &self.tools.schemas()).total();
        self.stats.push_turn(usage);
        if let Some(turn) = self.stats.turns.last() {
            self.debug_log(format!(
                "turn_complete sent={} would_be={} history_saved={} context_tokens={} session_sent={} session_would_be={} session_history_saved={}",
                turn.sent,
                turn.would_be,
                turn.saved,
                turn.context_tokens,
                self.stats.session_sent,
                self.stats.session_would_be,
                self.stats.session_saved
            ));
        }
        trace.final_assistant = assistant_text;
        if hit_tool_round_limit {
            return Err(LlmError::ToolRoundLimit { rounds: max_rounds });
        }
        Ok(())
    }

    async fn update_turn_scratchpad(
        &mut self,
        scratchpad: &TurnScratchpad,
        consumed_tool_batch: &[ChatMessage],
        assistant_response: &ChatMessage,
        usage: &mut TurnUsage,
    ) -> Result<TurnScratchpad, LlmError> {
        let source = turn_scratchpad_update_source(consumed_tool_batch, assistant_response);
        let messages = build_fitted_turn_scratchpad_update_messages(
            &scratchpad.summary,
            &source,
            self.config.context.max_tokens,
        );
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
        let response = self
            .client
            .complete_chat(ChatCompletionRequest {
                model: String::new(),
                messages,
                tools: Vec::new(),
                tool_choice: None,
                stream: false,
            })
            .await?;
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
            estimate_text_tokens(&summary)
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
            estimate_text_tokens(&summary)
        ));
        Ok(TurnScratchpad { summary })
    }

    async fn execute_tool_call(
        &mut self,
        call: &ToolCall,
    ) -> Result<ToolResult, crate::tools::ToolError> {
        let input = if call.function.arguments.trim().is_empty() {
            serde_json::Value::Object(Default::default())
        } else {
            serde_json::from_str(&call.function.arguments).map_err(|error| {
                crate::tools::ToolError::InvalidInput {
                    tool: call.function.name.clone(),
                    message: error.to_string(),
                }
            })?
        };
        let result = self.tools.execute(&call.function.name, input).await?;
        if result.refresh_manifest {
            self.manifest = MachineManifest::scan(&self.skills, &self.mcp);
        }
        Ok(result)
    }

    fn debug_log(&mut self, event: impl AsRef<str>) {
        if !self.debug_enabled {
            return;
        }
        self.debug
            .push(format!("[{}] {}", unix_timestamp_millis(), event.as_ref()));
    }
}

fn dedupe_images(images: &mut Vec<ImageAttachment>) {
    let mut seen = BTreeSet::new();
    images.retain(|image| seen.insert((image.source.clone(), image.base64_data.clone())));
}

fn exchange_user_input(text: &str, image_count: usize) -> String {
    if image_count == 0 {
        text.to_string()
    } else if text.trim().is_empty() {
        format!(
            "[{image_count} image{} attached]",
            if image_count == 1 { "" } else { "s" }
        )
    } else {
        format!(
            "{} [{} image{} attached]",
            text,
            image_count,
            if image_count == 1 { "" } else { "s" }
        )
    }
}

fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_validation_rejects_duplicate_case_ids() {
        let suite = EvalSuite {
            name: "sample".to_string(),
            default_model: None,
            cases: vec![
                EvalCase {
                    id: "same".to_string(),
                    description: None,
                    prompt: "one".to_string(),
                    model: None,
                    max_turns: None,
                    assertions: vec![EvalAssertion::AssistantContains {
                        value: "ok".to_string(),
                    }],
                },
                EvalCase {
                    id: "same".to_string(),
                    description: None,
                    prompt: "two".to_string(),
                    model: None,
                    max_turns: None,
                    assertions: vec![EvalAssertion::AssistantContains {
                        value: "ok".to_string(),
                    }],
                },
            ],
        };

        let error = validate_suite(&suite).unwrap_err();

        assert!(error.to_string().contains("duplicate eval case id"));
    }

    #[tokio::test]
    async fn assistant_contains_assertion_checks_final_text() {
        let trace = sample_trace("hello from eval", Vec::new());
        let models = ModelRegistry::default();
        let outcome = evaluate_assertion(
            &EvalAssertion::AssistantContains {
                value: "hello".to_string(),
            },
            &trace,
            &models,
            &trace.model,
        )
        .await;

        assert!(outcome.passed);
    }

    #[tokio::test]
    async fn tool_called_at_least_assertion_counts_matching_calls() {
        let calls = vec![
            tool_call("call_1", "read_file"),
            tool_call("call_2", "read_file"),
            tool_call("call_3", "batch"),
        ];
        let trace = sample_trace("done", calls);
        let models = ModelRegistry::default();
        let outcome = evaluate_assertion(
            &EvalAssertion::ToolCalledAtLeast {
                name: "read_file".to_string(),
                count: 2,
            },
            &trace,
            &models,
            &trace.model,
        )
        .await;

        assert!(outcome.passed, "{}", outcome.message);
    }

    #[test]
    fn next_turn_context_compacts_large_latest_tool_output_to_fit_budget() {
        let base = vec![
            ChatMessage::system("system"),
            ChatMessage::user("read large"),
        ];
        let scratchpad = TurnScratchpad {
            summary: "- large file contained marker".to_string(),
        };
        let tool_batch = vec![
            ChatMessage::assistant_tool_calls(
                String::new(),
                vec![tool_call("call_read", "read_file")],
            ),
            ChatMessage::tool("call_read", "large output ".repeat(2000)),
        ];

        let next = prepare_next_turn_context(&base, &scratchpad, &tool_batch, &[], 900).unwrap();

        assert!(next.preparation.before_tokens > next.preparation.after_tokens);
        assert!(next.preparation.after_tokens <= 900);
        let messages = build_turn_messages(&base, &next.scratchpad, &next.tool_batch);
        let tool_output = messages
            .iter()
            .find(|message| message.role == "tool")
            .and_then(ChatMessage::content_text)
            .unwrap();
        assert!(tool_output.chars().count() < 2500);
    }

    fn sample_trace(final_assistant: &str, tool_calls: Vec<ToolCall>) -> EvalTrace {
        EvalTrace {
            case: EvalCase {
                id: "sample".to_string(),
                description: None,
                prompt: "sample".to_string(),
                model: None,
                max_turns: None,
                assertions: Vec::new(),
            },
            model: ModelProfile {
                name: "fake".to_string(),
                base_url: "http://127.0.0.1".to_string(),
                model: "fake".to_string(),
                api_key: String::new(),
            },
            passed: false,
            duration_ms: 0,
            final_assistant: final_assistant.to_string(),
            requests: Vec::new(),
            events: Vec::new(),
            tool_calls,
            tool_results: Vec::new(),
            assertions: Vec::new(),
            stats: TokenLedger::default(),
            debug: Vec::new(),
            error: None,
        }
    }

    fn tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            kind: "function".to_string(),
            function: crate::llm::types::ToolCallFunction {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }
}
