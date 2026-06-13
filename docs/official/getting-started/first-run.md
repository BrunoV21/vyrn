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
cargo run -- --models
```

If `vyrn` is installed from Cargo later, use:

```bash
vyrn --models
```

Expected session shape:

```text
vyrn small context first
model llama3  context 4096

> inspect this package and suggest the next implementation step
• I will read the package files and summarize the current state...
tokens sent: 812 | saved: 3,204 | session saved: 11,847 | context: 1,024/4,096
```

Use `/exit` to close the session.

Useful local commands inside the session:

```text
/models     switch model profile
/stats      show token usage
/manifest   print compact machine manifest
/refresh    rescan manifest
/skills     list discovered skills
/clear      reset session context and clear the terminal
/exit       close the session
```
