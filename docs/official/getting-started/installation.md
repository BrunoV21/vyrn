# Installation

vyrn is currently scoped as a Rust package.

## From source

```bash
git clone https://github.com/BrunoV21/vyrn.git
cd vyrn
cargo build
cargo test
```

Run the package locally:

```bash
cargo run -- --help
```

Run the full test suite:

```bash
cargo test
```

Run the deterministic REPL end-to-end test:

```bash
cargo test --test e2e_repl -- --nocapture
```

The E2E test starts a local fake OpenAI-compatible streaming server, creates a temporary
model profile, pipes input into the real `vyrn` binary, and verifies a tool call through
the REPL.

## From Cargo

Once published:

```bash
cargo install vyrn
vyrn --help
```

## Model endpoint

vyrn expects an OpenAI-compatible chat completions API. Local endpoints commonly use an empty API key:

```toml
[models.local]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
api_key = ""
```

Hosted OpenAI-compatible services can use the same structure with their base URL and API key.
