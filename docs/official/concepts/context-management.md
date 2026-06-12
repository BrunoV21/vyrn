# Context Management

Context management is the core differentiator of vyrn.

## Rolling summary

vyrn does not send full conversation history on every request. It keeps a living summary that is rewritten at the start of each new user request.

```text
1. User sends a new request.
2. vyrn asks the model to update the current summary from the last exchange.
3. The updated summary replaces the old summary.
4. vyrn sends system prompt + summary + new request to the model.
5. The agent streams, uses tools, and completes the task.
```

Two model calls per user request are intentional. vyrn targets local and small models where token pressure matters and additional calls can be cheaper than large prompt reuse.

## What summaries preserve

- The user's high-level session goal.
- Decisions already made.
- File paths touched.
- Important outputs that still affect the task.
- Current constraints and open risks.

## What summaries drop

- Raw tool output once acted on.
- Intermediate reasoning.
- Repeated context.
- Old details that no longer affect the task.

## Aggressiveness

| Level | Behavior |
|---|---|
| `low` | Summarize older turns but keep recent tool results. |
| `medium` | Drop tool results from turns older than the latest one. |
| `high` | Drop all tool results and keep summary only. |

When the context budget is tight, vyrn should escalate pruning automatically.
