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

## UI contract

After each completed request, show a compact line:

```text
ok  tokens sent: 812 | saved: 3,204 | session total saved: 11,847
```

The stats line should be visible by default. It is part of how users understand that vyrn is behaving differently from large-context agents.

## Verbose mode

`vyrn --verbose` can show detailed token counts and raw summary information. The normal session UI should remain concise.
