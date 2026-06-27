# Agent evals

`vyrn eval` runs JSON-defined live agent tests against configured model profiles.
Use it to repeat development tasks, check tool behavior, and inspect detailed traces
when a model response regresses.

```sh
vyrn eval evals/basic.json
```

From a source checkout:

```sh
cargo run -- eval evals/basic.json
```

## Suite format

Keep committed suites under `evals/`.

```json
{
  "name": "core-agent",
  "default_model": "llama3",
  "cases": [
    {
      "id": "read-fixture",
      "prompt": "Read fixture.txt and tell me what it says.",
      "assertions": [
        { "type": "assistant_contains", "value": "hello from eval" },
        { "type": "tool_called", "name": "read_file" },
        { "type": "file_contains", "path": "fixture.txt", "value": "hello from eval" }
      ]
    }
  ]
}
```

## Commands

```sh
vyrn eval evals/basic.json
vyrn eval evals/basic.json --case read-fixture
vyrn eval evals/basic.json --model llama3
vyrn eval evals/basic.json --output .vyrn/eval-runs/latest
vyrn eval evals/basic.json --json
vyrn eval evals/basic.json --dry-run
vyrn eval evals/basic.json --no-debug
```

Eval cases run in the current repository by default. They can call normal vyrn
tools, including file edits and shell commands, so keep fixtures small and review
suite prompts before running them.

## Assertions

Supported assertion types:

- `assistant_contains`
- `assistant_not_contains`
- `tool_called`
- `tool_called_at_least`
- `tool_not_called`
- `file_exists`
- `file_contains`
- `command_succeeds`
- `judge`

Use deterministic assertions first. Add `judge` only when a semantic check cannot be
expressed with output, tool, file, or command assertions.

## Traces

Each run writes traces under `.vyrn/eval-runs/<timestamp>/` unless `--output` is
provided.

- `summary.json` records suite status, timing, and token totals.
- `<case-id>/trace.json` records requests, responses, tool calls, tool results,
  assertions, token accounting, and errors.
- `<case-id>/transcript.md` is the readable debugging transcript.
- `<case-id>/debug.log` contains debug events by default. Use `--no-debug` only
  when compact traces are preferred.

## Context Warnings

vyrn fits chained tool-call requests before sending them. Large tool outputs are
compacted into the turn scratchpad, and multiple tool calls from one assistant
response are processed incrementally so one large batch does not exceed the context
budget.

`batch` trims very large stdout/stderr before returning tool results. The trace keeps
the trimmed content plus a marker showing how much output was omitted.

Known limitations to watch in eval traces:

- The first model request can still exceed tiny context windows before any tool
  compaction is possible.
- Image payloads can be large and are not deeply budget-fitted yet.
- Provider tokenizers can differ from vyrn's local estimator.
- Future eager MCP schemas may increase tool schema size enough to require separate
  schema-budget handling.
