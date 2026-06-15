# Slash Commands

Slash commands are available inside an active interactive session.

In a real terminal, type `/` and press `Tab` to autocomplete slash commands.

## `/models`

Switch the active model profile without leaving the session. Use Up/Down and
Enter to choose a configured profile.

```text
/models
```

The selected model is stored as the last selected model for future sessions. `/model`
is kept as an alias.

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

In verbose mode, `/stats` also includes per-call accounting and per-call
contributors for each turn.

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
