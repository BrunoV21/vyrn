# Roadmap

The roadmap follows the phases in `docs/prd.md`.

## Phase 1 - Core Agent

| Version | Deliverable |
|---|---|
| v0.1 | Implemented: interactive REPL loop, streaming output, OpenAI client, `read_file`, `write_file`, `edit_file`. |
| v0.2 | Implemented: `batch` tool, machine manifest, `refresh_manifest`. |
| v0.3 | Implemented: rolling summary context manager and token savings tracking. |
| v0.4 | Implemented: Agent Skills discovery for `.agents/skills/`, `.vyrn/skills/`, and `~/.vyrn/skills/`. |
| v0.5 | Implemented: `--models`, `models.toml`, last-selected model startup, and in-session `/models`. |

## Phase 2 - MCP Integration

| Version | Deliverable |
|---|---|
| v0.6 | Partially implemented: `.mcp.json` metadata loading and merge precedence. Next: eager server tool loading. |
| v0.7 | Discovery mode with `list_mcp_tools`. |
| v0.8 | Implemented early: MCP config merging from `.agents/` and `.vyrn/`. Next: runtime activation behavior. |

## Non-goals for v1

- GUI or web interface.
- Cloud-hosted inference.
- Anthropic-specific APIs or features.
- Built-in RAG or vector search.
- Multi-agent orchestration.
