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
a filterable slash-command palette with `Tab` acceptance, and provides inline `/models`
selection. When stdin or stdout is not a TTY, it falls back to the plain text
prompt for scripts and tests.
During a turn, the agent can call `ask_user` to request clarification. In a real
terminal, vyrn renders selectable options plus an always-available freeform
reply. In plain text mode, type an option number or any freeform answer.
Vision-capable models can receive images in the current message through `Ctrl+V`
clipboard image paste or by mentioning image file paths in the prompt.
While a turn is running, press `Esc` to cancel it and return to the composer. In
the composer, press Up/Down to recall previous non-command prompts. To redirect
an active turn, type a message while the agent is working and press `Enter`.
vyrn interrupts the current model or tool wait, preserves work that already
completed, marks unrun tool calls as interrupted, and sends the steering message
before the agent chooses its next action.
Type `/`, press `Ctrl+O`, or press `F1` to open the in-session command palette.

The startup UI shows the boxed `vyrn` banner, selected model, and context budget.
If no model is selected at startup, vyrn uses the last selected model, then the
configured default, then the first configured model.
If no model profiles exist, vyrn starts a setup flow and writes the new profile to
`~/.vyrn/models.toml`.

## `vyrn --models`

Lists configured model profiles from `~/.vyrn/models.toml` plus any project-local
override file and lets the user pick one with Up/Down and Enter. The picker also
includes `configure new model`, which saves a new profile to `~/.vyrn/models.toml`.
`--model` is accepted as an alias.

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

For committed or shared model configuration, keep the secret in the environment:

```toml
[models.groq-fast]
base_url = "https://api.groq.com/openai/v1"
model = "llama-3.1-8b-instant"
api_key_env = "GROQ_API_KEY"
```

## `vyrn -p "prompt"`

Runs exactly one prompt without the startup banner or interactive `you:` prompt,
then exits. Use this for scripts and harnesses.

```bash
vyrn -p "inspect src and report the largest files"
vyrn -p "run the focused tests" --debug --context 2048
```

`-p` is the short form of `--prompt` and composes with the existing session
flags. With `--debug`, the one-shot run writes the same structured session JSON
as an interactive run, labeled with `run_kind: "prompt"` and a
`prompt_complete` or `prompt_error` end reason.

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
or non-2xx response body. Debug mode also writes structured LLM traces to
`.vyrn/debug/sessions/`. Each interactive session gets one JSON file, and `/clear`
starts a new trace file.

## `vyrn debug-viewer [trace.json]`

Writes a local static HTML viewer for debug trace JSON and prints its path.

```bash
vyrn debug-viewer
vyrn debug-viewer .vyrn/debug/sessions/1782857207560.json
```

Open the printed `viewer.html`, then load a session trace from
`.vyrn/debug/sessions/` or an eval case `llm-trace.json` file. The viewer is
local-only and uses a browser file picker; it does not start a server.

Running `vyrn debug-viewer` also creates `.vyrn/debug/sessions/` and
`.vyrn/eval-runs/` if they do not exist. The generated page shows those default
locations and embeds recent trace file paths when available. Browsers do not allow
static pages to force the native file picker to start in a specific directory, so
use the displayed paths or drag a trace JSON file into the page.

When a trace path is passed, vyrn embeds that JSON into the generated page and
renders it immediately. The viewer places human/agent interactions in one lane
and harness internals such as summary refreshes and turn scratchpads in another.
Each call shows provider usage, separately labeled estimates, context available,
and per-message estimates. Per-message values are estimates because providers
only return aggregate request usage.
