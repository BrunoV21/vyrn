# Skills and MCP

vyrn uses progressive disclosure for capabilities outside the core toolset.

## Skills

Skills follow the Agent Skills protocol.

Discovery paths, in priority order:

1. `.vyrn/skills/`
2. `~/.vyrn/skills/`
3. `.agents/skills/`

Global config can also live under `~/.vyrn/`.

At startup, vyrn keeps skill names in the compact manifest and adds an
`[available_skills]` system prompt section with each skill's source,
`SKILL.md` path, and description. Full `SKILL.md` activation remains part of the
progressive-disclosure workflow.

## Skill format

```text
.vyrn/skills/
└── my-skill/
    ├── SKILL.md
    ├── scripts/
    ├── references/
    └── assets/
```

```markdown
---
name: my-skill
description: What this skill does and when to use it.
---

# Instructions
...
```

## MCP

vyrn follows `.mcp.json` conventions. The current implementation parses server
metadata and shows eager or lazy mode in the compact manifest. Executing MCP servers
and loading MCP tool schemas is Phase 2 work.

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

Current behavior:

- `.agents/.mcp.json` and `.vyrn/.mcp.json` are loaded if present.
- `.vyrn/.mcp.json` takes precedence by server name.
- Servers render in the manifest as `name(eager)` or `name(lazy)`.

Planned Phase 2 behavior: `eager: true` loads server tools into the prompt, while
`eager: false` exposes compact server metadata until the agent discovers tools.
