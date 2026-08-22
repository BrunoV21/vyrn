# Getting Started

vyrn is designed for terminal-native users running local or small LLMs through an OpenAI-compatible API.

## Requirements

- Rust toolchain with Cargo.
- An OpenAI-compatible chat completions endpoint.
- A configured model profile in `~/.vyrn/models.toml`. Project-local
  `.vyrn/models.toml` is available only when a repository needs an override.

## Core flow

1. Configure one or more model profiles.
2. Run `vyrn init` in a repository when you want a local `.vyrn/` scaffold.
3. Start `vyrn` or `vyrn --models`.
4. Let the startup manifest report available binaries, skills, and MCP servers.
5. Work in the interactive REPL.
6. Watch the composer status row after each completed request.

## Minimal model profile

```bash
mkdir -p ~/.vyrn
```

```toml
# ~/.vyrn/models.toml
[models.llama3]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
api_key = ""
```

## Next

- [Install vyrn](./installation.md)
- [Run the first session](./first-run.md)
