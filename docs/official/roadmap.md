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

## Future candidates

These candidates should stay consistent with vyrn's product thesis: small
default context, local-first operation, OpenAI-compatible endpoints, and
deliberate capability loading.

### Deliberate memory

Persist useful facts without turning memory into always-loaded context.

vyrn could support project memory in `.vyrn/memory/` and global memory in
`~/.vyrn/memory/`. Each scope would keep a compact `index.json` containing brief
entries only:

```json
[
  {
    "id": "project-style",
    "scope": "project",
    "summary": "Use terse, terminal-native docs examples and avoid hosted surfaces.",
    "path": ".vyrn/memory/project-style.md",
    "tags": ["docs", "voice"],
    "updated_at": "2026-06-15"
  }
]
```

The index can be cheap enough to inspect during the summary update call. That
call becomes the decision point for memory: the model can decide whether a memory
entry is relevant, whether a new memory candidate should be written, or whether
an old entry should be ignored. Full memory bodies stay out of the prompt unless
the agent deliberately reads the referenced file.

Good first behavior:

- Load only memory indexes by default, never full memory bodies.
- Let the summary update produce `memory_to_load`, `memory_to_write`, and
  `memory_to_ignore` candidates.
- Keep project memory higher priority than global memory.
- Expose terminal-native controls such as `/memory`, `/remember`, and `/forget`.
- Track memory tokens separately in `/stats` so users can see when memory is
  helping or wasting context.

This is not RAG or vector search. It is explicit, file-backed, inspectable
memory for recurring preferences, project constraints, and decisions that should
survive session compaction.

### Context budget planner

Make context allocation visible and adaptive.

Instead of treating the prompt as one shared pile, vyrn could reserve rough
budgets for system text, rolling summary, memory index entries, active skills,
MCP tool schemas, and recent user turns. When pressure rises, it can reduce the
least valuable bucket first and explain the decision in verbose mode.

### Skill token budgets

Show the cost of activated skills before they become a hidden prompt expense.

The skills loader could estimate each `SKILL.md` footprint, surface that in
`/skills`, and warn when activating a large skill would crowd out summary or
memory. Large skills could include a compact quick-start section and deeper
references loaded only when needed.

### Session snapshots

Persist resumable session state without saving raw transcripts by default.

A snapshot could store the rolling summary, selected model, token totals,
active skills, active MCP tools, and memory decisions. This would let users
resume work while preserving vyrn's compact-context behavior.

### Model capability probes

Help small and local models choose the right operating mode.

An optional probe could test whether a configured model reliably supports tool
calling, JSON-shaped responses, long summaries, and streaming. vyrn can then
choose simpler prompts or stricter output contracts for weaker models.

### Local trace exports

Make token savings and context decisions auditable.

`vyrn --trace` could write a local run report with per-call token accounting,
summary revisions, activated skills, memory reads, and tool result sizes. The
normal UI stays compact, while advanced users can inspect why context changed.

## Non-goals for v1

- GUI or web interface.
- Cloud-hosted inference.
- Anthropic-specific APIs or features.
- Built-in RAG or vector search.
- Multi-agent orchestration.
