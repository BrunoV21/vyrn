<div align="center">
  <img src="docs/branding/assets/vyrn-banner.svg" alt="vyrn - token-efficient Rust CLI agent for local and small LLMs" width="100%" />

  <p><strong>Build for the smallest viable context first.</strong></p>

  <p>
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-8A2BE2?style=for-the-badge">
    <a href="https://brunov21.github.io/vyrn/"><img alt="Official documentation" src="https://img.shields.io/badge/docs-official-8A2BE2?style=for-the-badge"></a>
    <img alt="Rust 2024" src="https://img.shields.io/badge/rust-2024-00E5FF?style=for-the-badge&logo=rust&logoColor=white">
    <img alt="Package: Rust" src="https://img.shields.io/badge/package-rust-A970FF?style=for-the-badge&logo=rust&logoColor=white">
    <img alt="OpenAI compatible" src="https://img.shields.io/badge/api-OpenAI%20compatible-111111?style=for-the-badge">
  </p>
</div>

vyrn is a token-efficient, model-agnostic CLI agent built in Rust for developers and terminal-native users running local or small LLMs. It keeps the always-loaded prompt and tool surface tiny, uses raw shell batching as the main power primitive, and tracks token savings as a first-class product signal.

```text
┌──────────────┐   compact prompt    ┌──────────────┐
│ User / TTY   │ ──────────────────> │ vyrn agent   │
│              │ <────────────────── │ stream +     │
│ local task   │  answer + stats     │ token stats  │
└──────────────┘                     └──────┬───────┘
                                            │
                            OpenAI-compatible chat API
                                            │
                                     ┌──────▼───────┐
                                     │ local/small   │
                                     │ LLM endpoint  │
                                     └──────────────┘
```

## Why vyrn?

- **Small context first:** the system prompt, core tools, manifest, and history strategy are designed for tight context windows.
- **Model-agnostic:** any OpenAI-compatible endpoint can work, including Ollama, LM Studio, Groq, Together AI, OpenRouter, or a custom local server.
- **Raw power primitive:** `batch` is the main extension point for shell work, installs, scripts, and host inspection.
- **Rolling summaries:** conversation history is compressed into a living summary instead of resent wholesale on every turn.
- **Visible savings:** each completed request reports tokens sent, tokens saved, and session total saved.
- **Open standards:** skills follow Agent Skills protocol, and MCP config follows `.mcp.json` conventions.

## Installation

For local development, use a checkout:

```bash
git clone https://github.com/BrunoV21/vyrn.git
cd vyrn
cargo build
cargo test
cargo run -- --help
```

Once published, the Rust package is intended to install with Cargo:

```bash
cargo install vyrn
vyrn
```

## Quick Start

Create a model profile for an OpenAI-compatible local endpoint:

```toml
# .vyrn/models.toml
[models.llama3]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
api_key = ""
```

Start an interactive session:

```bash
vyrn --models
```

Expected session shape:

```text
> using llama3 @ localhost:11434
> manifest: git, curl, node, python3
> context budget: 4096 tokens

you: refactor the auth module to use JWT
vyrn: I'll inspect the current auth code first...
ok  tokens sent: 812  |  saved: 3,204  |  session total saved: 11,847
```

## Development

Run the Rust package from source:

```bash
cargo build
cargo test
cargo run -- --models
```

The current product requirements live in [`docs/prd.md`](docs/prd.md). Keep implementation, CLI behavior, and documentation aligned with that file until the product scope changes.

## Documentation

Official docs live in [`docs/official`](docs/official).
Agents should use the raw Markdown index: [`docs/official/agents.md`](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/agents.md).
Brand positioning and story notes live in [`docs/branding`](docs/branding).

```bash
cd docs/official
npm install
npm run docs:dev
```

The docs use a terminal-brutalist standard: black surfaces, violet brand/action states, cyan technical accents, and compact agent-readable pages.

## Project Status

vyrn is early Rust package infrastructure for a terminal-native, token-efficient agent. The current focus is the core REPL loop, OpenAI-compatible streaming, minimal tools, rolling context management, skills loading, and visible token savings.

## License

MIT
