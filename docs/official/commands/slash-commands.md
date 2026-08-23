# Slash Commands

Slash commands are available inside an active interactive session.

In a real terminal, type `/`, press `Ctrl+O`, or press `F1` to open the command
palette. Continue typing to filter it, use Up/Down to select, press `Tab` to
accept without running, or press `Enter` to run the selected completion. A left
click runs a visible command, and the mouse wheel changes the selection while
the pointer is over the palette. Resizing the terminal redraws the palette and
keeps the selected command inside its visible window.

## `/help`

List every in-session command and the core keyboard controls.

## `/models`

Switch the active model profile without leaving the session. Use Up/Down and
Enter to choose a configured profile, or select `configure new model` to add a
profile through the same setup flow used when vyrn starts without any configured
models.

```text
/models
```

The selected model is stored as the last selected model for future sessions. New
profiles are saved to `~/.vyrn/models.toml`. `/model` is kept as an alias.

## `/stats`

Print full token usage for the current session, including the largest estimated
token contributors.

```text
/stats
```

The compact status row still updates after each completed request. `/stats` adds a
ranked contributor list for system prompt text, rolling summaries, summary
input, summary output, user requests, images, skill metadata and loaded skill
files, tool schemas, tool call input, tool call output, assistant context,
assistant output, and message overhead.

Provider-reported prompt and completion counts are used whenever available.
Fallback counts, retained context, the raw-history counterfactual, and savings
are labeled as estimates.

In verbose mode, `/stats` also includes per-call accounting and per-call
contributors for each turn.

## `/context`

Show estimated context used and still available, rolling-summary and raw-history
size, provider-reported session tokens, fallback estimates, and estimated
history savings.

## `/summary`

Show the current model-generated rolling summary and its estimated retained
prompt footprint. In the full-screen TUI, the interaction-level summary controls
open the exact rolling summary that was supplied to each turn; the first
interaction explicitly reports that no previous exchange existed.

## `/scratchpad`

Show the latest compact scratchpad produced while tools were running in the last
turn. This exposes the task facts and outcomes vyrn retained as old tool batches
were ingested.

## `/manifest`

Print the current compact machine manifest.

```text
/manifest
```

The manifest should include available binaries, discovered skills, and MCP servers in compact form.

## `/refresh`

Trigger `refresh_manifest` manually.

```text
/refresh
```

Use this after installing tools or changing project skill/MCP configuration.

## `/skills`

List discovered skills with their source and `SKILL.md` path.

```text
/skills
```

Use this to see whether a skill came from project `.vyrn`, global `~/.vyrn`, or
project `.agents`, and which file the agent can read when activating it.

## `/debug`

Show whether structured tracing is enabled. When active, this prints the exact
`.vyrn/debug/sessions/<session-id>.json` path.

## `/clear`

Reset the session summary/history and clear the terminal UI.

```text
/clear
```

This should not delete config files, skills, or model profiles.
It also resets the in-memory token ledger for the current session.

## `/exit`

Exit the current session.

```text
/exit
```
