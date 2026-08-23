# Context Management

Context management is the core differentiator of vyrn.

## Rolling summary

vyrn does not send full conversation history on every request. It keeps a living summary that is rewritten at the start of each new user request.

```text
1. User sends a new request.
2. vyrn asks the model to update the current summary from the last exchange.
3. The updated summary replaces the old summary.
4. vyrn sends the bounded meaningful-goal anchor, updated summary, authoritative
   bounded tool memory, compact exact recent exchange, and new request to the model.
5. The agent streams, uses tools, and completes the task.
```

Each non-initial request adds one rolling-summary model call. Tool-turn
checkpoints are deterministic and do not require another inference call, which
keeps local-model latency and token use predictable.

## What summaries preserve

- The user's high-level session goal.
- Decisions already made.
- File paths touched.
- Important outputs that still affect the task.
- Current constraints and open risks.

The rolling summary is not the only memory source. vyrn ignores greetings when
selecting the first meaningful goal, keeps bounded verbatim anchors for that
goal and the most recent exchange, and accumulates an authoritative bounded
tool checkpoint. Checkpoints retain exact tool arguments and head-and-tail
result excerpts, plus a compact count of completed tools. Result evidence and
tool inputs have separate bounds, so severe pressure drops repeatable inputs
before exact result edges. Empty or capped summary responses use a deterministic
exchange fallback instead of silently erasing previous context.

Within a tool chain, vyrn starts pruning at 70% of the configured context
budget. This leaves headroom for provider-tokenizer differences and the next
model response instead of waiting until the estimated prompt already reaches
the hard limit. If reaching that target would erase all retained result
evidence, vyrn keeps a fact-preserving candidate below the hard limit instead.

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
