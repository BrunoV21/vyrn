# Session Options

## `vyrn`

Starts the interactive REPL using the configured default model.

```bash
vyrn
```

The startup output should show the selected model, compact machine manifest, available skills, and context budget.

## `vyrn --models`

Lists configured model profiles from `models.toml` and prompts the user to select one for the session.

```bash
vyrn --models
```

Model profiles can point at local or hosted OpenAI-compatible endpoints:

```toml
[models.groq-fast]
base_url = "https://api.groq.com/openai/v1"
model = "llama-3.1-8b-instant"
api_key = "gsk_..."
```

## `vyrn --context 2048`

Overrides the configured context budget for this session.

```bash
vyrn --context 2048
```

This should influence rolling summary aggressiveness and the available budget for prompt, manifest, skills, tools, and current user request.

## `vyrn --verbose`

Shows full token accounting and raw summary details.

```bash
vyrn --verbose
```

Verbose mode is for debugging context behavior. The default UI should stay compact.
