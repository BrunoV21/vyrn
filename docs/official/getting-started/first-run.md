# First Run

Create a local model profile:

```bash
mkdir -p .vyrn
```

```toml
# .vyrn/models.toml
[models.llama3]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
api_key = ""
```

Start with model selection:

```bash
vyrn --models
```

Expected session shape:

```text
> using llama3 @ localhost:11434
> manifest: git, curl, cargo
> context budget: 4096 tokens

you: inspect this package and suggest the next implementation step
vyrn: I will read the package files and summarize the current state...
ok  tokens sent: 812 | saved: 3,204 | session total saved: 11,847
```

Use `/exit` to close the session.
