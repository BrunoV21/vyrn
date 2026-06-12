# Core Tools

Core tool descriptions must stay short. Token cost is part of the API.

## Always-loaded tools

| Tool | Purpose |
|---|---|
| `read_file` | Read a file at a path. |
| `write_file` | Write content to a file, creating it if needed. |
| `edit_file` | Replace an exact string in a file. |
| `batch` | Execute shell commands on the host. |
| `refresh_manifest` | Rescan the host and replace the compact manifest. |

## `batch`

`batch` is the primary power primitive. Anything that does not deserve an always-loaded dedicated tool should happen through shell commands.

```json
{
  "name": "batch",
  "commands": [
    "cargo test",
    "rg \"TODO\" src",
    "curl -s http://localhost:11434/v1/models"
  ]
}
```

If one command fails, later commands may still run. The model is responsible for choosing safe command sequences.

## Tool design rule

A new core tool must earn its permanent prompt cost. If a behavior is rare, project-specific, or easily expressed through shell commands, keep it out of the core tool list.
