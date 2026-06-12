# Skills and MCP

vyrn uses progressive disclosure for capabilities outside the core toolset.

## Skills

Skills follow the Agent Skills protocol.

Discovery paths, in priority order:

1. `.vyrn/skills/`
2. `.agents/skills/`

Global config can also live under `~/.vyrn/`.

At startup, vyrn should load only skill names and descriptions into the compact manifest. Full `SKILL.md` content is loaded only when the task matches that skill.

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

vyrn follows `.mcp.json` conventions and supports eager or discovery mode per server.

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

`eager: true` loads all tools into the prompt at startup. `eager: false` exposes only compact server metadata until the agent asks to discover tools.
