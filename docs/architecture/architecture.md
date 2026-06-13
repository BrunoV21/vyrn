# vyrn Architecture

Developer architecture for implementing the requirements in [`docs/prd.md`](../prd.md).

This document is intentionally implementation-facing. User-facing concepts belong in
`docs/official/`; this file describes the internal Rust structure, runtime flows, data
contracts, and phase plan needed to build vyrn.

## Product Constraints

vyrn is a Rust CLI package for terminal-native agent sessions against
OpenAI-compatible chat completion APIs. The architecture must preserve these
constraints:

- Keep startup prompt, tool descriptions, and manifest extremely small.
- Treat token budget and token savings as product behavior, not diagnostics.
- Use the OpenAI `/v1/chat/completions` schema and streaming protocol only.
- Support local and small hosted models without vendor-specific assumptions.
- Prefer one powerful raw shell primitive, `batch`, over many permanent tools.
- Let capability grow through progressive disclosure: skills first, MCP later.
- Avoid GUI, hosted inference, RAG, multi-agent orchestration, and proprietary APIs.

## Crate Shape

Start as a single binary crate with internal modules. Split into workspace crates only
after APIs stabilize or integration tests become expensive.

```text
vyrn/
|-- Cargo.toml
|-- src/
|   |-- main.rs
|   |-- cli.rs
|   |-- app.rs
|   |-- config/
|   |   |-- mod.rs
|   |   |-- models.rs
|   |   `-- paths.rs
|   |-- llm/
|   |   |-- mod.rs
|   |   |-- openai.rs
|   |   |-- stream.rs
|   |   `-- types.rs
|   |-- agent/
|   |   |-- mod.rs
|   |   |-- loop.rs
|   |   |-- prompt.rs
|   |   |-- context.rs
|   |   |-- tokens.rs
|   |   `-- transcript.rs
|   |-- tools/
|   |   |-- mod.rs
|   |   |-- core.rs
|   |   |-- batch.rs
|   |   |-- file.rs
|   |   `-- manifest.rs
|   |-- skills/
|   |   |-- mod.rs
|   |   |-- discovery.rs
|   |   `-- skill.rs
|   |-- mcp/
|   |   |-- mod.rs
|   |   `-- config.rs
|   |-- tui/
|   |   |-- mod.rs
|   |   |-- repl.rs
|   |   `-- render.rs
|   `-- error.rs
`-- tests/
    |-- config_loading.rs
    |-- context_manager.rs
    |-- tool_executor.rs
    `-- manifest.rs
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `main` | Install panic/error handling, parse CLI args, start `App`. |
| `cli` | `clap` definitions for startup flags and command help. |
| `app` | Compose config, model profile, manifest, client, tools, and REPL. |
| `config` | Load and merge `.vyrn` and `~/.vyrn` configuration. |
| `llm` | OpenAI-compatible request/response models and streaming client. |
| `agent` | Agent loop, prompt assembly, rolling summary, token stats. |
| `tools` | Core tool registry and host execution backends. |
| `skills` | Agent Skills discovery and activation. |
| `mcp` | `.mcp.json` loading and later MCP client runtime. |
| `tui` | Terminal input/output, streaming rendering, slash commands. |
| `error` | Shared typed errors and user-readable formatting. |

## Runtime Composition

`App` owns long-lived session dependencies:

```rust
pub struct App {
    config: EffectiveConfig,
    model: ModelProfile,
    client: OpenAiClient,
    tools: ToolRegistry,
    manifest: MachineManifest,
    skills: SkillRegistry,
    context: ContextManager,
    stats: TokenLedger,
}
```

The REPL should mutate only the session state that changes between turns:

- active model profile
- rolling summary
- latest exchange transcript
- token ledger
- current machine manifest
- activated skills and, in phase 2, activated MCP tools

Configuration structs should be immutable after startup except when `/models` changes
the active model profile or `/refresh` replaces the manifest.

## Startup Flow

```text
main
  -> parse CLI args
  -> discover project root from current working directory
  -> load config.toml and models.toml
  -> resolve active model
  -> discover skills
  -> load .mcp.json metadata
  -> scan machine manifest
  -> build compact system prompt
  -> start REPL
```

Startup should fail fast for missing model profiles and malformed config. Missing
optional paths such as `.agents/skills/`, `.vyrn/skills/`, and `.mcp.json` are normal.

## Configuration Model

### Files

```text
<project>/.vyrn/config.toml
<project>/.vyrn/models.toml
<project>/.agents/.mcp.json
<project>/.vyrn/.mcp.json
~/.vyrn/config.toml
~/.vyrn/models.toml
~/.vyrn/skills/
```

The PRD states that global vyrn-specific settings override local settings. Implement
merging with this precedence:

1. built-in defaults
2. project `.vyrn/config.toml`
3. global `~/.vyrn/config.toml`
4. CLI flags

Model profiles use the same precedence, with profile names as map keys. If the same
profile exists locally and globally, the global profile wins. CLI model selection
chooses a profile; it should not rewrite config.

### Effective Config

```rust
pub struct EffectiveConfig {
    pub context: ContextConfig,
    pub agent: AgentConfig,
    pub manifest: ManifestConfig,
}

pub struct ContextConfig {
    pub max_tokens: usize,
    pub summary_aggressiveness: SummaryAggressiveness,
}

pub struct AgentConfig {
    pub default_model: String,
    pub stream: bool,
}

pub struct ManifestConfig {
    pub auto_refresh: bool,
}
```

Use `serde` for TOML/JSON parsing and keep validation separate from deserialization so
error messages can name the bad field and source path.

## OpenAI-Compatible LLM Client

The LLM layer should expose a small interface that hides HTTP details but does not hide
OpenAI concepts:

```rust
#[async_trait]
pub trait ChatClient {
    async fn stream_chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatStream, LlmError>;

    async fn complete_chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError>;
}
```

Use `complete_chat` for summary refreshes and `stream_chat` for agent responses.
Streaming should parse server-sent events and surface normalized chunks:

```rust
pub enum StreamEvent {
    TextDelta(String),
    ToolCallDelta(ToolCallDelta),
    ToolCallDone(ToolCall),
    Finished(FinishReason),
}
```

Do not add provider-specific options in v1. If a provider requires compatibility
quirks, put them behind a narrow `CompatibilityMode` enum only after an actual failing
endpoint requires it.

## Prompt Architecture

The system prompt has four compact sections:

```text
[role] terminal coding agent. conserve tokens.
[rules] use tools when needed. prefer batch for shell. summarize aggressively.
[tools] read_file, write_file, edit_file, batch, refresh_manifest
[env] git,curl,node,python3 | skills:code-review | mcp:filesystem(eager)
```

Implementation should generate prompt text from structured inputs instead of storing
one large handwritten prompt. This keeps token-cost reviews straightforward.

Prompt assembly inputs:

- static base instructions
- compact tool descriptors
- current machine manifest
- activated skill instructions, if any
- eager MCP tool descriptors, phase 2
- rolling summary
- current user request

The core prompt builder should expose a debug method for `--verbose`:

```rust
pub struct PromptBundle {
    pub system: String,
    pub messages: Vec<ChatMessage>,
    pub estimated_tokens: TokenEstimate,
}
```

## Agent Loop

Each user request follows this sequence:

```text
1. Read user input.
2. Handle slash command if input starts with `/`.
3. Refresh rolling summary from previous exchange, unless this is the first turn.
4. Build chat request from system prompt, summary, active tools, and new request.
5. Stream assistant output.
6. Accumulate tool calls.
7. Execute requested tools.
8. Send tool results back to the model until it returns a final answer.
9. Persist the latest exchange into transcript memory.
10. Print token stats.
```

Tool calls may require multiple model round trips in one user turn. Token accounting
must include every call in the completed request: summary call, assistant call, and
tool-result follow-up calls.

### Turn State

```rust
pub struct TurnState {
    pub user_input: String,
    pub assistant_text: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub usage: TurnUsage,
}
```

Store raw tool output only in the latest exchange. Older tool output should be
available through the rolling summary or discarded according to pruning policy.

## Context Management

`ContextManager` owns:

- current rolling summary
- previous raw exchange
- configured aggressiveness
- context budget
- token estimator

```rust
pub struct ContextManager {
    summary: Option<String>,
    previous_exchange: Option<Exchange>,
    policy: SummaryPolicy,
    max_tokens: usize,
}
```

Before each non-initial user turn, ask the model to rewrite the summary:

```text
Update the session summary for the next turn.
Keep the user's original high-level goal, active constraints, decisions,
paths touched, and unresolved tasks.
Drop raw tool output that is no longer needed.
Aggressiveness: medium.
```

The summary request should not include the full historical transcript. It includes:

- existing summary
- raw previous user input
- raw previous assistant final text
- previous tool calls and results, filtered by current aggressiveness

### Aggressiveness Escalation

Start with the configured level. Escalate automatically when estimated prompt size
approaches budget:

| Condition | Effective level |
|---|---|
| under 70% of budget | configured level |
| 70-90% of budget | at least `medium` |
| over 90% of budget | `high` |

Escalation is per turn and should not rewrite the user's stored config.

## Token Accounting

Track token usage at three levels:

```rust
pub struct TokenLedger {
    pub session_sent: usize,
    pub session_would_be: usize,
    pub session_saved: isize,
    pub turns: Vec<TurnUsage>,
}

pub struct TurnUsage {
    pub sent: usize,
    pub would_be: usize,
    pub saved: isize,
    pub calls: Vec<CallUsage>,
}
```

`sent` is measured from the actual request payload when possible. If the API returns
usage, prefer provider usage for completed non-streaming calls and fall back to local
estimation for streaming calls or providers that omit usage.

`would_be` is an estimate of sending the full transcript without summary pruning:

```text
system prompt + manifest + active tools + full raw transcript + current request
```

For local estimation, use a tokenizer crate when available for the configured model
family. If no tokenizer is known, use a documented heuristic. Keep the estimator
replaceable because tokenization differs across OpenAI-compatible endpoints.

Update the composer status row after each completed request:

```text
tokens sent: 812 | saved: 3,204 | session saved: 11,847 | context: 1,024/4,096
```

In `--verbose`, print per-call usage and the current summary.

## Tool System

Core tools should be represented by one trait and a compact JSON schema generator:

```rust
#[async_trait]
pub trait Tool {
    fn name(&self) -> &'static str;
    fn compact_description(&self) -> &'static str;
    fn json_schema(&self) -> serde_json::Value;
    async fn execute(&self, input: serde_json::Value) -> Result<ToolResult, ToolError>;
}
```

Always-loaded tools:

| Tool | Backend |
|---|---|
| `read_file` | `tokio::fs::read_to_string` with size guard. |
| `write_file` | parent directory creation plus atomic-ish write. |
| `edit_file` | exact single-string replacement with mismatch error. |
| `batch` | sequential shell commands using the user's shell. |
| `refresh_manifest` | rescan binaries, skills, and MCP metadata. |

### File Tool Rules

- Paths are resolved relative to the process current working directory unless absolute.
- Do not silently create parent directories for `edit_file`.
- `edit_file` must fail if the old string is missing.
- `edit_file` must fail if the old string appears more than once unless the input
  explicitly allows replacing all occurrences in a later version.
- File reads should have a default byte limit to avoid dumping huge files into context.

### Batch Tool Rules

`batch` executes commands in order and returns stdout/stderr/status per command.
Failures do not stop later commands.

```rust
pub struct BatchInput {
    pub commands: Vec<String>,
}

pub struct BatchCommandResult {
    pub command: String,
    pub status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}
```

Use the user's default shell from the environment. Add a per-command timeout in config
only if needed; for v0.2 a hardcoded conservative timeout is acceptable.

The PRD explicitly says no sandboxing. Do not add pretend safety layers that change the
contract. Errors should be clear and raw enough for the model to act on.

## Machine Manifest

The manifest is a compact startup snapshot, not a full environment inventory.

```rust
pub struct MachineManifest {
    pub binaries: Vec<String>,
    pub skills: Vec<SkillSummary>,
    pub mcp_servers: Vec<McpServerSummary>,
}
```

Scan only known binaries with `which`/PATH lookup:

```text
git,curl,wget,python3,node,npm,pnpm,cargo,rustc,docker,chrome,ffmpeg,rg
```

Do not run version commands during startup. Manifest rendering should target roughly
20-40 tokens:

```text
[env] git,curl,node,python3,cargo,rg
[skills] code-review,deploy-check
[mcp] filesystem(eager),postgres(lazy)
```

`refresh_manifest` replaces the current manifest in memory. It must not append another
manifest section to the prompt.

## Skills Architecture

Implement Agent Skills progressive disclosure in two layers:

1. Discovery: read only frontmatter `name` and `description`.
2. Activation: read full `SKILL.md` and optionally referenced bundled files.

Discovery paths in priority order:

```text
<project>/.vyrn/skills/
~/.vyrn/skills/
<project>/.agents/skills/
```

If two skills share a name, keep the higher-priority one and surface a verbose warning.

```rust
pub struct SkillSummary {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
    pub source: SkillSource,
}

pub struct ActivatedSkill {
    pub summary: SkillSummary,
    pub instructions: String,
}
```

Skill activation can initially be model-driven: the model sees the names/descriptions
in the manifest and asks `read_file` for the relevant `SKILL.md`. A later dedicated
`activate_skill` tool may be added only if this proves too verbose or error-prone.

## MCP Architecture

Phase 2 should keep MCP metadata separate from executable tools until a server is
activated.

Config loading:

```text
<project>/.agents/.mcp.json
<project>/.vyrn/.mcp.json
```

`.vyrn` entries take precedence over `.agents` entries by server name.

```rust
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub eager: bool,
}
```

Phase 2 milestones:

- v0.6: parse `.mcp.json`, start eager servers, include eager tool schemas.
- v0.7: expose lazy servers in manifest and add `list_mcp_tools`.
- v0.8: complete config merging and selective activation.

Lazy MCP tool schemas must not be loaded into the base system prompt.

## TUI And Slash Commands

The terminal UI preserves native scrollback. TTY sessions use crossterm raw-mode input
for a styled composer and slash completion; piped sessions use a plain line-oriented
fallback.

Startup output:

```text
vyrn small context first
model llama3  context 4096
```

Slash commands are handled locally and never sent to the model:

| Command | Handler |
|---|---|
| `/models` | Select and switch active model profile. |
| `/stats` | Render `TokenLedger`. |
| `/manifest` | Print current manifest. |
| `/refresh` | Rescan manifest. |
| `/skills` | List discovered and activated skills. |
| `/clear` | Reset summary, previous exchange, and turn ledger. |
| `/exit` | End REPL. |

For `--models`, list model profile names, base URLs, and model IDs. Do not print API
keys.

## Error Handling

Use typed errors internally and concise messages externally.

```rust
pub enum VyrnError {
    Config(ConfigError),
    Llm(LlmError),
    Tool(ToolError),
    Io(std::io::Error),
}
```

Errors shown to users should include:

- what failed
- file path or tool name when relevant
- next action when obvious

Do not dump Rust backtraces unless `RUST_BACKTRACE` is set or a future debug flag is
enabled.

## Recommended Dependencies

Initial dependencies should stay small and conventional:

| Need | Candidate |
|---|---|
| CLI parsing | `clap` |
| async runtime | `tokio` |
| HTTP client | `reqwest` |
| serialization | `serde`, `serde_json`, `toml` |
| errors | `thiserror`, `anyhow` at app boundary |
| async traits | `async-trait` |
| streaming SSE | `eventsource-stream` or direct `reqwest` bytes parsing |
| directories | `directories` |
| terminal prompts | `dialoguer` or simple stdin first |
| native terminal input | `crossterm` |

The implementation uses a split frontend: real terminals use crossterm raw-mode input
while preserving native terminal scrollback, while non-TTY stdin/stdout keep the plain
text runner for tests and scripting.

## Testing Strategy

Unit tests:

- config precedence and validation
- model profile resolution
- prompt rendering token-conscious formatting
- summary aggressiveness escalation
- token savings calculations
- exact-match file editing
- batch command result collection
- manifest rendering and replacement
- skill frontmatter parsing and precedence
- `.mcp.json` merge behavior

Integration tests:

- fake OpenAI-compatible server for non-streaming summary calls
- fake streaming chat completion with text deltas
- fake tool-call loop with `read_file` and `edit_file`
- REPL slash command handlers without a real model

Current deterministic E2E coverage lives in `tests/e2e_repl.rs`. It starts a local
fake OpenAI-compatible streaming server, creates a temporary `.vyrn/models.toml`, pipes
input into the real `vyrn` binary, and verifies a model-requested `read_file` call
flows through the REPL.

Golden tests are appropriate for compact prompt and manifest output. Any prompt change
should make token impact visible in the diff.

## Delivery Plan

Follow the PRD phases, but build vertical slices that can run end to end.

### v0.1 Core REPL

- crate scaffolding
- CLI flags: `--help`, `--context`, `--verbose`
- config loading
- model profile loading
- OpenAI client with streaming text
- line-oriented REPL
- `read_file`, `write_file`, `edit_file`

### v0.2 Host Power

- `batch`
- startup machine manifest
- `refresh_manifest`
- `/manifest` and `/refresh`

### v0.3 Context And Savings

- rolling summary call
- summary aggressiveness
- local token estimation
- per-turn and session token savings
- `/stats`

### v0.4 Skills

- `.vyrn/skills` and `.agents/skills` discovery
- manifest skill summaries
- skill precedence
- activated skill injection path
- `/skills`

### v0.5 Model UX

- `--models`
- interactive profile selection
- `/models`
- last-selected model startup fallback
- model switch without dropping current summary

### v0.6-v0.8 MCP

- `.mcp.json` parsing
- eager MCP server tool loading
- lazy server discovery with `list_mcp_tools`
- `.agents` and `.vyrn` MCP merge precedence

## Architectural Risks

| Risk | Mitigation |
|---|---|
| Small models emit malformed tool calls. | Keep schemas simple, retry with compact correction, and show raw errors in verbose mode. |
| Token estimates differ by endpoint. | Treat local counts as estimates and prefer provider usage when returned. |
| Summary loses important task state. | Always preserve original goal, touched paths, decisions, and open tasks in the summary prompt. |
| `batch` output overwhelms context. | Display full output to terminal but summarize or truncate result payload sent back to the model. |
| MCP eager mode defeats token goals. | Make eager explicit per server and keep lazy as the default recommendation. |
| Config precedence surprises users. | Print effective model/context in startup output and detailed sources in verbose mode. |

## Implementation Rules

- A new always-loaded tool must justify its permanent prompt cost.
- Prompt strings should be generated from structured data and covered by golden tests.
- Slash commands are local control plane commands, not model messages.
- `refresh_manifest` replaces prompt state instead of appending history.
- Raw transcript is short-lived; rolling summary is the durable context primitive.
- User-facing docs must stay aligned with this architecture and the PRD, but detailed
  module internals should remain here.
