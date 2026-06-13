# Session Options

## `vyrn`

Starts the interactive REPL using the configured default model.

```bash
vyrn
```

From a source checkout:

```bash
cargo run --
```

In a real terminal, `vyrn` starts a styled native-scrollback chat UI with raw-mode
keyboard input. It keeps normal terminal scrollback, streams model output live, supports
Tab completion for slash commands, and provides inline `/models` selection. When stdin
or stdout is not a TTY, it falls back to the plain text prompt for scripts and tests.

The startup UI shows the boxed `vyrn` banner, selected model, and context budget.
If no model is selected at startup, vyrn uses the last selected model, then the
configured default, then the first configured model.

## `vyrn --models`

Lists configured model profiles from `models.toml` and prompts the user to select one for the session.

```bash
vyrn --models
```

From a source checkout:

```bash
cargo run -- --models
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

From a source checkout:

```bash
cargo run -- --context 2048
```

This should influence rolling summary aggressiveness and the available budget for prompt, manifest, skills, tools, and current user request.

## `vyrn --verbose`

Shows full token accounting and raw summary details.

```bash
vyrn --verbose
```

From a source checkout:

```bash
cargo run -- --verbose
```

Verbose mode is for debugging context behavior. The default UI should stay compact.

## `vyrn --debug`

Shows provider request details when errors occur.

```bash
vyrn --debug
```

From a source checkout:

```bash
cargo run -- --debug
```

Use this when a provider request fails and you need the request URL, network error kind,
or non-2xx response body.
