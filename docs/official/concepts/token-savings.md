# Token Savings

Token savings is a product feature, not just a diagnostic metric.

## Accounting

vyrn tracks token usage across each model call:

```text
tokens_sent      = actual tokens in the request
tokens_would_be  = estimated tokens if full history were sent
tokens_saved     = tokens_would_be - tokens_sent
```

The session total is the sum of saved tokens across completed requests.

`/stats` also ranks estimated prompt contributors for the current session:
system prompt text, summaries, user requests, images, skill metadata and loaded
skill files, tool schemas, tool call input, tool call output, assistant context,
and message overhead.

## UI contract

After each completed request, update the compact composer status row:

```text
tokens sent: 812 | saved: 3,204 | session saved: 11,847 | context: 1,024/4,096
```

The status row should be visible by default in the typing zone. It is part of how users understand that vyrn is behaving differently from large-context agents.

The `context` value is the estimated current prompt footprint compared with the
configured context budget.

## Verbose mode

`vyrn --verbose` can show detailed per-call token counts, per-call contributor
breakdowns, and raw summary information. The normal session UI should remain
concise.
