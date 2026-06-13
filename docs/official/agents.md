# vyrn Raw Docs for Agents

This page is a raw Markdown index for agents. Prefer these links when you need documentation without rendered HTML, navigation scripts, or theme markup.

## Primary Entry Points

- [Getting started](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/getting-started/index.md)
- [First run](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/getting-started/first-run.md)
- [Installation](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/getting-started/installation.md)
- [Commands overview](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/commands/index.md)
- [Architecture](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/concepts/architecture.md)

## Install

Rust package from source:

```bash
git clone https://github.com/BrunoV21/vyrn.git
cd vyrn
cargo build
cargo test
cargo run -- --help
```

Deterministic end-to-end REPL test:

```bash
cargo test --test e2e_repl -- --nocapture
```

This starts a fake local OpenAI-compatible streaming server and verifies a real `vyrn`
binary session can execute a model-requested file tool call.

Published package shape:

```bash
cargo install vyrn
vyrn --models
```

- Package: `vyrn` on crates.io once published
- Installed command: `vyrn`

## Command Reference

- [Session options](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/commands/session.md)
- [Slash commands](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/commands/slash-commands.md)

## Concepts

- [Architecture](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/concepts/architecture.md)
- [Context management](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/concepts/context-management.md)
- [Core tools](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/concepts/core-tools.md)
- [Skills and MCP](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/concepts/skills-and-mcp.md)
- [Token savings](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/concepts/token-savings.md)

## Project Docs

- [Community](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/community.md)
- [Roadmap](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/roadmap.md)
- [Release notes](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/releases/index.md)

## Recommended Agent Reading Order

1. Read [Architecture](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/concepts/architecture.md).
2. Read [Core tools](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/concepts/core-tools.md).
3. Read [Context management](https://raw.githubusercontent.com/BrunoV21/vyrn/main/docs/official/concepts/context-management.md).
4. Run `vyrn --help` or `vyrn <flag> --help` for the installed CLI's local contract.

## Important Runtime Contract

- vyrn is an interactive terminal agent, not a GUI or hosted service.
- Keep the always-loaded tool surface minimal.
- Prefer `batch` for host work that does not need a dedicated compact tool.
- Preserve the user's original high-level session goal in summaries.
- Show token savings in the composer status row after each completed request.
- Treat MCP runtime tool execution as Phase 2; current code parses MCP metadata for the manifest.
