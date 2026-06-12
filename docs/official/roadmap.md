# Roadmap

The roadmap follows the phases in `docs/prd.md`.

## Phase 1 - Core Agent

| Version | Deliverable |
|---|---|
| v0.1 | Interactive REPL loop, streaming TUI, OpenAI client, `read_file`, `write_file`, `edit_file`. |
| v0.2 | `batch` tool, machine manifest, `refresh_manifest`. |
| v0.3 | Rolling summary context manager and token savings tracking. |
| v0.4 | Agent Skills loader for `.agents/skills/` and `.vyrn/skills/`. |
| v0.5 | `--models`, `models.toml`, and in-session `/model`. |

## Phase 2 - MCP Integration

| Version | Deliverable |
|---|---|
| v0.6 | `.mcp.json` loading and eager mode. |
| v0.7 | Discovery mode with `list_mcp_tools`. |
| v0.8 | MCP config merging from `.agents/` and `.vyrn/`. |

## Non-goals for v1

- GUI or web interface.
- Cloud-hosted inference.
- Anthropic-specific APIs or features.
- Built-in RAG or vector search.
- Multi-agent orchestration.
