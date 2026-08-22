# Terminal Coding Agent Pain Points and the vyrn Product Gap

Research snapshot: 2026-08-15.

## Purpose

This document summarizes recurring pain points in terminal coding agents such as
Claude Code, Codex CLI, Pi, and OpenCode. It then defines the specific gap vyrn
is intended to address and maps that gap to concrete product choices.

This is positioning and product-direction research, not a claim that every
capability described below has shipped. Current commitments remain defined by
[`docs/prd.md`](../prd.md), and implementation details remain defined by
[`docs/architecture/architecture.md`](../architecture/architecture.md).

## Executive Conclusion

The central market gap is not a lack of agent features. It is a lack of a small,
predictable, inspectable control loop for developers who cannot assume a large
hosted model, a large context window, or an unlimited token budget.

The leading terminal agents are increasingly capable, but capability often
arrives with:

- large fixed context overhead;
- lossy or opaque compaction;
- approval fatigue or overly broad access;
- provider-specific behavior hidden behind compatibility claims;
- fragile long-running process handling;
- terminal rendering problems in real operator environments; and
- configuration, session state, and cost that are difficult to explain.

vyrn is intended to solve this by treating context, task state, execution, and
cost as bounded and visible resources. It should remain useful with a local or
small OpenAI-compatible model where a heavier harness becomes slow, expensive,
or unreliable.

> **Product gap:** terminal users need an agent whose operating state can be
> understood at every turn, and whose useful capability does not depend on a
> large hidden prompt or a large proprietary context window.

## Evidence Baseline

A 2026 empirical study manually classified 3,864 public bug reports from Claude
Code, Codex CLI, and Gemini CLI. API and integration errors accounted for 21.4%
of root causes, configuration and setup for 15.9%, user interaction and UI for
11.1%, and platform compatibility for 10.5%. AI logic and behavior accounted
for 10%. The most common symptoms were API errors, terminal problems, and
command failures.

The important implication is that model intelligence alone does not determine
the quality of a coding agent. The harness around the model is a substantial
part of the product.

Source: [Engineering Pitfalls in AI Coding Tools: An Empirical Study of Bugs in
Claude Code, Codex, and Gemini CLI](https://www.eecs.yorku.ca/~wangsong/papers/fse26-industry.pdf).

Public issue reports do not measure incidence across products: user bases,
reporting practices, and release velocity differ. Individual reports may also
describe versions that have since been fixed. They are used here to identify
recurring failure modes, not to rank products by raw issue count.

## Recurring Pain Points

### 1. Context Is Expensive Before Work Begins

System instructions, built-in tool schemas, MCP schemas, skills, repository
rules, file contents, screenshots, and command results all consume the same
finite context as the user's task.

A Claude Code report measured seven MCP servers consuming 67,300 tokens before
meaningful work began. Claude Code now documents deferred MCP tool definitions
as the default, which validates progressive disclosure as a product direction,
although reports still describe transports for which large tool sets were
loaded eagerly.

Sources:

- [Lazy-load MCP tool definitions to reduce context usage](https://github.com/anthropics/claude-code/issues/11364)
- [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)
- [HTTP MCP tools loaded eagerly](https://github.com/anthropics/claude-code/issues/40314)

The user-visible cost is shorter useful sessions, slower prompt processing,
earlier compaction, and quota consumption that appears disproportionate to the
request.

### 2. Compaction Can Preserve Conversation but Lose the Task

Summarizing a transcript is not the same as preserving operational state. A
summary can omit the current objective, a mandatory constraint, the next step,
or which validation remains incomplete.

Reported failure modes include sessions that cannot be resumed because they
are already over the context limit, resumed sessions that repeat completed
work, and compaction loops that consume credits without reducing the prompt.

Sources:

- [Claude Code session cannot compact before resume](https://github.com/anthropics/claude-code/issues/14472)
- [Codex loses task intent after resume](https://github.com/openai/codex/issues/8310)
- [OpenCode context overflow kills the session](https://github.com/anomalyco/opencode/issues/33376)
- [OpenCode infinite compaction loop](https://github.com/anomalyco/opencode/issues/27924)
- [Pi compaction design](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/compaction.md)

The core pain is uncertainty: the user cannot easily see what survived and
whether the agent is still solving the same task.

### 3. Permissions Trade Interruption for Risk

Conservative permission modes can interrupt routine work repeatedly. Broad
permission modes eliminate the interruption but weaken the boundary between a
repository task and the rest of the machine.

Reports across harnesses include remembered approvals not being honored,
permission prompts appearing in bypass or full-access modes, modal prompts
stealing keystrokes from an in-progress message, and shell syntax bypassing
path-based restrictions.

Sources:

- [Claude Code bypass mode still prompts](https://github.com/anthropics/claude-code/issues/34923)
- [Codex approvals not remembered](https://github.com/openai/codex/issues/15884)
- [Codex approval prompts steal composer input](https://github.com/openai/codex/issues/31801)
- [OpenCode shell redirection bypasses the external-directory permission](https://github.com/anomalyco/opencode/issues/32628)
- [Pi permission and sandbox extension examples](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/examples/extensions/README.md)

Parsing arbitrary shell commands into reliable fine-grained permission rules is
inherently difficult. Users need a small, understandable trust model more than
an ever-growing collection of command patterns.

### 4. OpenAI-Compatible Does Not Mean Agent-Compatible

OpenAI-compatible providers disagree on details that matter to an agent loop:

- `system` versus `developer` roles;
- Chat Completions versus Responses semantics;
- streamed tool-call encoding;
- tool-only assistant messages;
- reasoning parameters;
- parallel tool calls;
- finish reasons;
- usage fields; and
- model context metadata.

OpenCode documents support for more than 75 providers and local models, but
reports include blank responses, unrecognized streamed tool calls, unsupported
reasoning fields, and local endpoints that work directly but hang through the
harness. Pi exposes explicit compatibility switches such as
`supportsDeveloperRole` and `supportsReasoningEffort`, demonstrating that a
base URL and model ID do not fully describe a provider.

Sources:

- [OpenCode provider documentation](https://dev.opencode.ai/docs/providers)
- [OpenCode custom provider returns an empty response](https://github.com/anomalyco/opencode/issues/15756)
- [OpenCode fails to recognize Ollama tool calls](https://github.com/anomalyco/opencode/issues/20995)
- [Pi custom model compatibility](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/models.md)

This is especially important for local and small models, which are more likely
to expose partial implementations and stricter tool-call behavior.

### 5. Shell Invocation Is Not Process Supervision

An agent must manage operating-system processes, not merely invoke a shell.
Reliable execution requires process groups, bounded output, cancellation,
timeouts, background handles, exit notification, and clear failure states.

Reported failures include timeouts that leave child processes alive, turns that
finish while background commands still run, background completion that does
not wake the agent, and verbose output that causes excessive memory use.

Sources:

- [Codex timeout leaves child processes alive](https://github.com/openai/codex/issues/4337)
- [Codex turn completes before background work](https://github.com/openai/codex/issues/14731)
- [Codex background completion requires polling](https://github.com/openai/codex/issues/32188)
- [OpenCode unbounded command output and resource failure](https://github.com/anomalyco/opencode/issues/13230)

For local-model users, repeated polling is particularly wasteful because it
adds prompt processing and model turns to a task that should be event-driven.

### 6. Terminal UI Reliability Is Part of Agent Reliability

The difficult environments are the normal environments for terminal-native
users: tmux, SSH, IDE terminals, narrow panes, resized panes, long streams, and
large scrollback histories.

Recurring issues include broken native scrollback, cursor and focus
instability, whole-screen flicker, transcript scrolling competing with prompt
input, and session history consuming excessive UI memory.

Sources:

- [Claude Code captures tmux scrolling](https://github.com/anthropics/claude-code/issues/38810)
- [Codex cursor instability during streaming](https://github.com/openai/codex/issues/11063)
- [Pi streaming transcript flicker](https://github.com/earendil-works/pi/issues/3371)
- [OpenCode long-session TUI memory growth](https://github.com/anomalyco/opencode/issues/24445)

A coding task may be progressing correctly while the interface makes it appear
stuck or destroys the user's ability to inspect prior output. That is a product
failure, not cosmetic polish.

### 7. Effective State, Configuration, and Cost Are Hard to Explain

Users need to know which configuration and instruction files were loaded, why a
setting won, which task constraints remain active, what occupies the context
window, and how fast the session is consuming it.

Codex reports describe `AGENTS.md` being loaded but applied inconsistently in
later turns. OpenCode reports describe undocumented configuration locations and
unexpected precedence. Users of Claude Code and Codex have separately requested
first-class usage and context breakdowns.

Sources:

- [Codex applies AGENTS.md inconsistently](https://github.com/openai/codex/issues/25884)
- [OpenCode configuration location ambiguity](https://github.com/anomalyco/opencode/issues/18953)
- [OpenCode nested configuration precedence](https://github.com/anomalyco/opencode/issues/21307)
- [Claude Code usage analytics request](https://github.com/anthropics/claude-code/issues/33978)
- [Codex context breakdown request](https://github.com/openai/codex/issues/27898)

The useful metrics are not just `used / maximum`. They include fixed overhead,
growth per turn, tool-result contribution, compaction count, retained task
state, and the savings available from starting a clean session.

## Harness Trade-Offs

| Harness | Primary attraction | Characteristic trade-off |
| --- | --- | --- |
| Claude Code | A tightly integrated Claude experience with a broad extension surface | Context and quota pressure, permission friction, terminal regressions, and vendor dependence |
| Codex CLI | Strong OpenAI coding-model integration, sandboxing, and long-horizon autonomy | Approval friction, session recovery, background-process lifecycle, and TUI complexity |
| Pi | A minimal prompt, model choice, session-tree concepts, and deep extensibility | Safety and richer workflows are assembled through extensions; provider quirks still require configuration |
| OpenCode | Broad provider and local-model support with a flexible client/server architecture | A large provider-adapter matrix, configuration complexity, permission edge cases, and heavier runtime state |

A community benchmark found that Pi had the smallest prompt and least total
context traffic in its comparison. On one MCP-heavy task, Pi used 81% less total
input and 53% fewer model requests than OpenCode. This result is directional,
not a neutral industry benchmark, but it shows that harness design can
materially change context cost.

Source: [Pi, OpenCode, and Codex token-overhead benchmark](https://github.com/earendil-works/pi/discussions/6646).

## The Gap vyrn Is Solving

The market has strong answers for users who want the most capable proprietary
model, the widest extension ecosystem, or the largest provider catalog. It has
a weaker answer for a different user:

> A terminal-native developer running a local or inexpensive
> OpenAI-compatible model with a small context window, who values predictable
> execution, explicit state, and low overhead more than feature density.

For this user, a feature is not free merely because it is not invoked. Its
schema, instructions, runtime state, rendering, and failure paths can still
consume tokens and operator attention.

vyrn therefore competes on four properties:

1. **Bounded context** — every always-loaded component must justify its token
   cost.
2. **Durable task state** — the current objective and constraints must survive
   compaction independently of conversational prose.
3. **Predictable execution** — commands must have observable lifecycle and
   bounded output.
4. **Explainability** — the user must be able to inspect effective context,
   configuration, model capabilities, and session state.

The concise positioning is:

> **vyrn is the small-context coding agent whose state and cost stay visible.**

## How vyrn Addresses the Gap

### Product Direction Already Defined

The current PRD establishes several correct foundations:

- a Rust-only, terminal-native package;
- a tiny system prompt and minimal core tool set;
- a compact machine manifest;
- OpenAI-compatible local and hosted endpoints;
- skills loaded through progressive disclosure;
- lazy MCP discovery rather than eager schema injection;
- rolling context management; and
- token savings as a first-class product metric.

These choices directly target fixed prompt overhead, runtime weight, and
unnecessary capability loading.

### Requirements Needed to Complete the Promise

The research suggests that token efficiency alone is not enough. The following
requirements should guide implementation and future PRD revisions.

#### Separate Task State from Conversation Summary

Maintain a compact deterministic checkpoint containing:

- original objective;
- active user and repository constraints;
- completed and pending steps;
- files changed;
- validation already run; and
- the next expected action.

Rolling summaries may remain model-generated, but the checkpoint must not
depend on the model remembering to preserve these fields. Context reduction
should never make the active task pointer implicit.

#### Probe Provider Capabilities

When a model profile is configured, vyrn should be able to detect or explicitly
record:

- accepted message roles;
- streaming tool-call support;
- reasoning-parameter support;
- parallel tool-call behavior;
- tool-only message requirements;
- usage reporting; and
- effective context and output limits.

The resulting capability manifest should be small and cached. A failed
capability should produce a specific diagnostic rather than an empty response
or stalled loop.

#### Make `batch` a Supervised Primitive

`batch` should remain the main power primitive, but its runtime must provide:

- process-group termination;
- configurable timeouts;
- bounded in-context output;
- complete output stored outside the conversation with a retrievable path;
- cancellation;
- structured exit status;
- background process handles; and
- event-driven completion.

The raw shell interface can stay small while the process supervisor remains
robust.

#### Use a Small Trust Model

The PRD currently describes raw host execution without sandboxing. That is
simple and token-efficient, but it leaves a major trust gap. vyrn should avoid
building a large shell-pattern permission language and instead expose a small
set of understandable execution profiles, for example:

- `read-only`;
- `project-write`; and
- `unrestricted`.

The active profile and effective workspace boundary must always be visible.
The exact sandbox mechanism can remain platform-dependent, but the behavior
must be explicit and testable.

#### Make Effective State Inspectable

The terminal surface should eventually answer four questions without asking
the model:

```text
/context   What is consuming the context budget?
/session   What task, constraints, and pending work are retained?
/config    Which files and values produced the effective configuration?
/doctor    What can this endpoint and model actually support?
```

These commands should be deterministic local diagnostics. They should not spend
tokens merely to explain runtime state that vyrn already owns.

#### Preserve Native Terminal Behavior

Prefer an inline, append-oriented interface that cooperates with native
scrollback. Rendering must be bounded by visible state rather than total
session size. A late modal must never consume text already entered in the
composer. tmux, SSH, narrow panes, terminal resizing, and long streamed output
should be treated as primary test environments.

#### Make Configuration Precedence Explainable

The current PRD gives global vyrn settings precedence over project settings.
That is deliberate but opposite the convention many developers expect. Whether
the rule remains or changes, `/config` must display the source of every
effective value and documentation must use one consistent precedence model.

## Product Priorities

The research supports the following order:

1. Preserve a genuinely small prompt and bounded tool output.
2. Make the agent loop and foreground command lifecycle reliable.
3. Keep deterministic task checkpoints across compaction and resume.
4. Expose context, session, provider, and configuration diagnostics.
5. Add a small, explicit execution trust model.
6. Expand skills and MCP only through progressive disclosure.

Features that expand the permanent system prompt or introduce hidden runtime
state should carry a measurable context budget and a clear user benefit.

## Non-Goals

This gap does not require vyrn to become a universal agent platform. In line
with the PRD, the response does not include:

- a GUI or web application;
- hosted inference;
- npm distribution or a JavaScript workspace;
- built-in RAG;
- proprietary-provider features; or
- multi-agent orchestration.

The opportunity is narrower and more defensible: make one local Rust CLI agent
exceptionally economical, predictable, and understandable under constrained
inference.

