# Commands

vyrn has a deliberately small command surface. The primary interface is the interactive REPL.

## CLI commands

| Command | Description |
|---|---|
| `vyrn` | Start an interactive session with the default model. |
| `vyrn --models` | Select a configured model profile before starting. |
| `vyrn --context 2048` | Override the context budget for this session. |
| `vyrn --verbose` | Show full token counts and raw summary information. |
| `vyrn --debug` | Show provider URLs, network details, and response bodies on errors. |

From a source checkout, prefix commands with `cargo run --`, for example:

```bash
cargo run -- --models
```

## In-session slash commands

Slash commands operate inside an active `vyrn` session.

| Command | Description |
|---|---|
| `/models` | Switch model profile mid-session. |
| `/stats` | Show full token usage for the session. |
| `/manifest` | Print the current machine manifest. |
| `/refresh` | Trigger `refresh_manifest` manually. |
| `/skills` | List discovered skills. |
| `/clear` | Reset session summary/history and clear the terminal UI. |
| `/exit` | Exit vyrn. |

## Design rule

Add new commands only when they justify their prompt and documentation cost. For broad host interaction, prefer the `batch` tool instead of expanding the always-loaded command surface.
