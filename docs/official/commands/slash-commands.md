# Slash Commands

Slash commands are available inside an active interactive session.

## `/model`

Switch the active model profile without leaving the session.

```text
/model
```

The session summary remains active unless the user also clears it.

## `/stats`

Print full token usage for the current session.

```text
/stats
```

The compact stats line still appears after each completed request.

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

List discovered skills.

```text
/skills
```

Only skill names and descriptions should be shown until a skill is activated.

## `/clear`

Reset the session summary and history.

```text
/clear
```

This should not delete config files, skills, or model profiles.

## `/exit`

Exit the current session.

```text
/exit
```
