# Getting Started

vyrn is designed for terminal-native users running local or small LLMs through an OpenAI-compatible API.

## Requirements

- Rust toolchain with Cargo.
- An OpenAI-compatible chat completions endpoint.
- A configured model profile in `.vyrn/models.toml` or `~/.vyrn/models.toml`.

## Core flow

1. Configure one or more model profiles.
2. Start `vyrn` or `vyrn --models`.
3. Let the startup manifest report available binaries, skills, and MCP servers.
4. Work in the interactive REPL.
5. Watch the token stats line after each completed request.

## Minimal model profile

```toml
[models.llama3]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
api_key = ""
```

## Next

- [Install vyrn](./installation.md)
- [Run the first session](./first-run.md)
