# vyrn — Product Requirements Document

> A token-efficient, model-agnostic CLI agent built in Rust for developers and terminal-native users running local or small LLMs.

---

## 1. Vision & Problem Statement

Most CLI agents — Claude Code, Codex, Aider — are designed for large, context-rich models. They front-load thousands of tokens into every session: full tool descriptions, all MCP server schemas, large system prompts. This makes them impractical for:

- Small, local LLMs with limited context windows (e.g. Ollama models, quantized variants)
- Users who want speed and low overhead over feature density
- Environments where token cost or latency matters

**vyrn** flips this assumption. It is designed from the ground up to minimize context usage at every layer — from the system prompt, to tool descriptions, to conversation history — while remaining fully capable through extensibility.

**Core thesis:** Build for the smallest viable context first. Let capability grow from there.

---

## 2. Goals

- Ship a terminal-native interactive agent optimised for small context windows
- Keep the core toolset minimal and the system prompt tiny
- Make the `batch` tool the primary power primitive — raw, flexible, fast
- Implement intelligent rolling context management that the model drives
- Be fully OpenAI API compatible — no vendor lock-in
- Follow open standards: Agent Skills protocol and `.mcp.json` conventions
- Track and surface token savings as a core product metric and differentiator

---

## 3. Non-Goals (v1)

- GUI or web interface
- Cloud-hosted inference
- Anthropic-specific APIs or features
- Built-in RAG or vector search
- Multi-agent orchestration

---

## 4. Target User

Terminal-native developers and technical users who are comfortable running tools via CLI. They are running local models (Ollama, LM Studio, etc.) or cheap API-compatible endpoints (Groq, Together, OpenRouter). They care about speed, token efficiency, and control.

---

## 5. Architecture Overview

```
vyrn/
├── Agent Loop (interactive REPL)
│   ├── Context Manager       ← rolling summary + pruning
│   ├── LLM Client            ← OpenAI-compatible, streaming
│   ├── Tool Executor         ← core tools + batch
│   └── TUI                   ← streaming output, token stats
│
├── Core Tools (always loaded, minimal token cost)
│   ├── read_file
│   ├── write_file
│   ├── edit_file
│   ├── batch                 ← raw shell executor
│   └── refresh_manifest      ← rescan machine, reinject
│
├── Machine Manifest          ← fast startup scan of available tools
│
├── Skills Loader             ← Agent Skills protocol (progressive disclosure)
│   ├── .agents/skills/
│   └── .vyrn/skills/
│
└── MCP Client (Phase 2)      ← .mcp.json, eager or discovery mode
```

---

## 6. Configuration

### 6.1 Directory Structure

vyrn follows a two-tier config system:

```
<project>/
├── .agents/
│   ├── skills/               ← shared Agent Skills (cross-agent)
│   └── .mcp.json             ← shared MCP config (cross-agent)
│
└── .vyrn/
    ├── config.toml           ← vyrn-specific settings (local)
    ├── skills/               ← vyrn-specific skills
    └── models.toml           ← configured model profiles
```

A global config can also live at `~/.vyrn/` — **global overrides local** for all vyrn-specific settings.

### 6.2 `config.toml`

```toml
[context]
max_tokens = 4096          # Context window budget
summary_aggressiveness = "medium"  # low | medium | high
                                   # high = drop all tool call results, summarize only

[agent]
default_model = "llama3"
stream = true

[manifest]
auto_refresh = false       # Only refresh when agent calls refresh_manifest
```

### 6.3 `models.toml`

Users define named model profiles:

```toml
[models.llama3]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
api_key = ""

[models.groq-fast]
base_url = "https://api.groq.com/openai/v1"
model = "llama-3.1-8b-instant"
api_key = "gsk_..."
```

### 6.4 `.mcp.json`

Follows the standard `.mcp.json` convention with an added `eager` field per server:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "eager": true
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "eager": false
    }
  }
}
```

- `eager: true` — all tools from this server are loaded into the system prompt at session start
- `eager: false` — the agent sees only the server name and a short description; it must call a `list_mcp_tools` tool to discover and selectively activate tools

vyrn loads `.mcp.json` from both `.agents/` and `.vyrn/` directories, merging them with `.vyrn/` taking precedence.

---

## 7. Core Tools

All tool descriptions in the system prompt must be as short as possible. Token cost is a first-class concern.

### 7.1 `read_file`
Read the contents of a file at a given path.

### 7.2 `write_file`
Write content to a file, creating it if it does not exist.

### 7.3 `edit_file`
Replace a specific string in a file with new content. Requires exact match.

### 7.4 `batch`
Execute one or more shell commands in sequence on the host machine. Raw passthrough — no sandboxing, no step structure. The model is responsible for safe usage.

```json
{
  "name": "batch",
  "commands": [
    "which chrome",
    "curl -s https://example.com | head -20",
    "pip install httpx"
  ]
}
```

Returns stdout and stderr per command. If a command fails, subsequent commands still run unless the model chooses to stop.

The `batch` tool is the primary extensibility primitive. Anything not in the core tools — downloading, installing, running scripts, interacting with browsers, system calls — goes through `batch`.

### 7.5 `refresh_manifest`
Rescan the host machine for available tools and reinject a compact manifest into the system prompt. Replaces the previous manifest — does not append. The agent calls this when it suspects the environment has changed (e.g. after installing a new tool via `batch`).

---

## 8. Machine Manifest

On startup, vyrn performs a fast scan of the host environment and produces a compact manifest injected into the system prompt. Designed to cost ~20–40 tokens maximum.

**What it checks:**
- Common binaries in PATH (git, curl, wget, python3, node, cargo, docker, chrome, ffmpeg, etc.)
- Active MCP servers (from `.mcp.json`)
- Available skills (names + descriptions only, from `.agents/skills/` and `.vyrn/skills/`)

**Example manifest injection:**
```
[env] git, curl, node, python3, chrome, docker
[skills] pdf-export, code-review, deploy-check
[mcp] filesystem(eager), postgres(lazy)
```

The manifest is fast because it only checks binary availability (`which`), not versions or configurations. Full rescanning is triggered by `refresh_manifest`.

---

## 9. Skills (Agent Skills Protocol)

vyrn implements the [Agent Skills](https://agentskills.io) open standard for extending agent capabilities with packaged knowledge and workflows.

**Reference:** https://agentskills.io/specification

### 9.1 Discovery Paths

vyrn discovers skills from two locations, in priority order:

1. `.vyrn/skills/` — vyrn-specific skills (local or global `~/.vyrn/skills/`)
2. `.agents/skills/` — shared cross-agent skills

If the same skill name exists in both, the `.vyrn/` version takes precedence.

### 9.2 Progressive Disclosure (per spec)

1. **Discovery** — at startup, load only `name` and `description` from each `SKILL.md` frontmatter. Injected into the manifest, not the full system prompt.
2. **Activation** — when a task matches, the agent reads the full `SKILL.md` into context.
3. **Execution** — the agent follows instructions, optionally loading bundled scripts or reference files.

### 9.3 Skill Format

Follows the Agent Skills specification exactly:

```
.vyrn/skills/
└── my-skill/
    ├── SKILL.md          # Required: name, description, instructions
    ├── scripts/          # Optional
    ├── references/       # Optional
    └── assets/           # Optional
```

```markdown
---
name: my-skill
description: What this skill does and when to use it.
---

# Instructions
...
```

---

## 10. Context Management

This is the core differentiator of vyrn.

### 10.1 The Rolling Summary

vyrn does not send the full conversation history on each request. Instead, it maintains a **rolling summary** — a living, compressed representation of what has happened so far that is rewritten at the start of each new user request.

**Flow per request:**

```
1. User sends new request
2. vyrn makes Call 1 → LLM:
   "Here is the current summary and the raw last exchange.
    Produce an updated summary that captures what is still
    relevant. Drop tool call results that are no longer needed."
3. Updated summary replaces old summary in context array
4. vyrn makes Call 2 → LLM:
   [system prompt] + [updated summary] + [new user request]
5. Agent responds, executes tools, completes task
```

Two LLM calls per request is intentional and acceptable — vyrn targets local models where inference is fast and there is no per-token billing.

### 10.2 What the Summary Preserves

The model decides what to keep. General heuristics it is instructed to follow:

- Keep: task goals, decisions made, file paths touched, important outputs
- Drop: raw tool call results once acted on, intermediate reasoning, repeated context
- Always keep: the user's original high-level goal for the session

### 10.3 Pruning Aggressiveness

Configured via `summary_aggressiveness` in `config.toml`:

| Level | Behaviour |
|---|---|
| `low` | Summarise old turns but keep recent tool results |
| `medium` | Drop tool results from turns older than the last one |
| `high` | Drop all tool call results entirely; summary only |

When the context budget is tight (approaching `max_tokens`), vyrn automatically escalates aggressiveness.

### 10.4 Token Savings Tracking

vyrn tracks token usage across every call:

```
tokens_sent      = actual tokens in the request (measured)
tokens_would_be  = tokens if full history were sent (estimated)
tokens_saved     = tokens_would_be - tokens_sent
```

This is tracked per-request and accumulated as a session total.

---

## 11. TUI & UX

### 11.1 Interactive REPL

vyrn runs as an interactive terminal session. The user enters requests, the agent responds, executes tools, and streams output — all within a single persistent session.

```
vyrn

> using llama3 @ localhost:11434
> manifest: git, curl, node, python3, chrome
> skills: pdf-export (1), code-review (2)
> context budget: 4096 tokens

you: refactor the auth module to use JWT

vyrn: I'll start by reading the current auth implementation...
[reading src/auth.rs]
...
```

### 11.2 Streaming

LLM responses stream token by token directly to the terminal as they are returned from the API. Tool calls and results are displayed inline as they execute.

### 11.3 Token Stats Display

After each completed request, vyrn displays a compact stats line:

```
✓  tokens sent: 812  |  saved: 3,204  |  session total saved: 11,847
```

This is a permanent fixture in the UI — not hidden, not optional. It is a core part of the product identity.

### 11.4 `--models` Flag

```bash
vyrn --models
```

Lists all configured model profiles from `models.toml` and prompts the user to select one for the session. Can also be invoked mid-session via a `/model` command.

### 11.5 CLI Commands

| Command | Description |
|---|---|
| `vyrn` | Start interactive session with default model |
| `vyrn --models` | Select model before starting |
| `vyrn --context 2048` | Override context budget for this session |
| `vyrn --verbose` | Show full token counts and raw summaries |

### 11.6 In-Session Slash Commands

| Command | Description |
|---|---|
| `/model` | Switch model mid-session |
| `/stats` | Show full token usage for the session |
| `/manifest` | Print current machine manifest |
| `/refresh` | Trigger `refresh_manifest` manually |
| `/skills` | List loaded skills |
| `/clear` | Reset session summary and history |
| `/exit` | Exit vyrn |

---

## 12. OpenAI API Compatibility

vyrn uses the OpenAI `/v1/chat/completions` API format exclusively. No Anthropic-specific APIs, no vendor-specific extensions.

This means vyrn works out of the box with:

- **Ollama** (local, `http://localhost:11434/v1`)
- **LM Studio** (local)
- **Groq** (fast inference API)
- **Together AI**
- **OpenRouter**
- **Any OpenAI-compatible endpoint**

Tool calling uses the standard OpenAI `tools` / `tool_choice` format.

---

## 13. Phased Delivery

### Phase 1 — Core Agent (v0.1–v0.5)

| Version | Deliverable |
|---|---|
| v0.1 | Interactive REPL loop, streaming TUI, OpenAI client, `read/write/edit_file` |
| v0.2 | `batch` tool, machine manifest, `refresh_manifest` |
| v0.3 | Rolling summary context manager, token savings tracking |
| v0.4 | Agent Skills loader (`.agents/skills/`, `.vyrn/skills/`) |
| v0.5 | `--models` flag, `models.toml`, in-session `/model` switch |

### Phase 2 — MCP Integration

| Version | Deliverable |
|---|---|
| v0.6 | `.mcp.json` loading, eager mode (all tools upfront) |
| v0.7 | Discovery mode (`list_mcp_tools` tool, selective activation) |
| v0.8 | MCP config merging from `.agents/` and `.vyrn/` |

---

## 14. Key Design Principles

1. **Token budget is a first-class constraint.** Every design decision — tool descriptions, system prompt, history management — is evaluated through the lens of token cost.
2. **The model drives context decisions.** vyrn does not mechanically truncate. It asks the model what to keep.
3. **Raw over structured.** The `batch` tool is a raw shell passthrough. Simplicity beats safety theatre.
4. **Two calls per request is fine.** Local models are fast and free. More calls for better quality is a good tradeoff.
5. **Open standards first.** Agent Skills, `.mcp.json`, OpenAI API — no proprietary lock-in anywhere.
6. **Token savings is a product feature, not a metric.** It should be visible, satisfying, and shareable.

---

## 15. Open Questions / Future Considerations

- **Manifest caching** — should the manifest be stored to disk and loaded on next startup to avoid even the fast scan?
- **Per-skill token budgeting** — should activating a large skill reduce the available summary budget?
- **Session persistence** — should the rolling summary be saved to disk so a session can be resumed?
- **vyrn hub / skill registry** — a public index of shareable community skills (post-v1)
- **Structured batch output** — an optional mode where `batch` returns per-command structured JSON for easier model reasoning (post-v1)

---

## 16. References

- Agent Skills specification: https://agentskills.io/specification
- Agent Skills home: https://agentskills.io/home
- OpenAI API reference: https://platform.openai.com/docs/api-reference
- MCP specification: https://modelcontextprotocol.io/specification