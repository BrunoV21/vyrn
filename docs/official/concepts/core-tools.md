# Core Tools

Core tool descriptions must stay short. Token cost is part of the API.

## Always-loaded tools

| Tool | Purpose |
|---|---|
| `read_file` | Read a file at a path. |
| `read_image` | Attach one or more image files to the current model turn. |
| `write_file` | Write content to a file, creating it if needed. |
| `edit_file` | Replace an exact string in a file. |
| `batch` | Execute shell commands on the host from the current working directory by default. |
| `refresh_manifest` | Rescan the host and replace the compact manifest. |

## `batch`

`batch` is the primary power primitive. Anything that does not deserve an always-loaded dedicated tool should happen through shell commands.
Commands run from the current working directory by default unless the command explicitly changes directories or uses an absolute path.

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

The current implementation returns structured per-command results with command,
exit status, stdout, stderr, and timeout state.

In the interactive TTY, completed tools render a compact preview of the result so the
user can see what happened without expanding the full tool payload.

## `read_image`

`read_image` accepts `paths` for multiple image files, or `path` for one file. Supported
types are `png`, `jpg`, `jpeg`, `webp`, and `gif`. Results are attached to the next
model round as OpenAI-compatible base64 image data URLs, while the transcript keeps only
a compact text note.

Use `read_image` for image files instead of shelling out to Python through `batch`.

## `edit_file`

`edit_file` replaces one exact string. It fails if the old string is missing or if
the old string appears more than once.

## Tool design rule

A new core tool must earn its permanent prompt cost. If a behavior is rare, project-specific, or easily expressed through shell commands, keep it out of the core tool list.
