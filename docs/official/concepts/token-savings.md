# Token Savings

Token savings is a product feature, not just a diagnostic metric.

## Accounting

vyrn tracks token usage across each model call:

```text
tokens_spent          = request input tokens plus known or estimated output tokens
tokens_would_be       = estimated tokens if prior raw history were sent
history_tokens_saved  = tokens_would_be - tokens_spent
```

The session total is the sum of history tokens saved across completed requests.
This metric is intentionally about prior conversation history: it compares the
rolling summary against sending the previous raw exchange history again. It does
not claim every possible saving from current-turn tool result truncation,
current-turn tool history compaction, or omitted intermediate terminal output.

`spent` is cumulative model I/O for all model calls in a turn. A multi-tool turn
can therefore spend far more tokens than the final `context` value, because
`context` is only the retained prompt footprint after the turn has finished.

`/stats` also ranks estimated token contributors for the current session:
system prompt text, rolling summaries, summary input, summary output, user
requests, images, skill metadata and loaded skill files, tool schemas, tool call
input, tool call output, assistant context, assistant output, and message
overhead.

## UI contract

After each completed request, update the compact composer status row:

```text
turn spent: 812 | turn history saved: 3,204 | session history saved: 11,847 | context: 1,024/4,096
```

The status row should be visible by default in the typing zone. It is part of how users understand that vyrn is behaving differently from large-context agents.

The `context` value is the estimated current prompt footprint compared with the
configured context budget.

## Verbose mode

`vyrn --verbose` can show detailed per-call token counts, per-call contributor
breakdowns, and raw summary information. The normal session UI should remain
concise.
