# vyrn Brand Story

vyrn starts from one constraint: most CLI agents assume they can spend thousands of tokens before the user even asks for work, and many now assume 128K+ context windows are a practical default. That is comfortable for large hosted models, but often the wrong fit for local and small LLMs running on constrained hardware.

vyrn chooses the opposite default.

It treats context like memory on a small machine: precious, visible, and worth managing carefully. Large always-loaded prompts increase memory pressure, slow turns down, and waste tokens before useful work starts. The agent starts with a tiny prompt, a compact machine manifest, a minimal tool surface, and a rolling summary that keeps only what matters.

The name **vyrn** is short, terminal-friendly, and intentionally lightweight. It should feel like a small binary that does a serious job without dragging a large runtime, hosted service, or proprietary stack behind it.

The product philosophy is:

> Build for the smallest viable context first. Let capability grow from there.

vyrn is not trying to be the biggest agent. It is trying to be the agent that remains useful when context is scarce, local inference matters, and the terminal is the interface.

## Tagline

**vyrn: build for the smallest viable context first.**

## Positioning

vyrn is a token-efficient Rust CLI agent for developers and terminal-native users running local or small OpenAI-compatible models.

## Palette

The visual system should reinforce constrained inference, measurable efficiency, and terminal-native utility.

- `violet` is the identity color: product name, primary actions, active states.
- `steel blue` is the technical color: model state, manifests, tools, system framing.
- `green` is reserved for efficiency and successful outcomes: token savings, confirmations, healthy execution.
- `amber` and `red` stay semantic: warnings and failures only.

Core palette:

- `#06070A` background
- `#0D1016` surface
- `#151A22` raised surface
- `#273142` border
- `#3A475E` border strong
- `#F3F7FB` primary text
- `#98A3B3` muted text
- `#677287` dim text
- `#8B5CF6` violet
- `#A78BFA` violet hover
- `#7DA2C2` steel blue
- `#A9BDD3` steel blue strong
- `#9FE870` efficiency green
- `#F5A524` amber
- `#F43F5E` red

## Voice

- Practical over grandiose.
- Compact over verbose.
- Local-first, model-agnostic, and debuggable.
- Rust-native without assuming the reader already knows Rust.
- Honest about tradeoffs: fewer default tools, more deliberate capability loading.

## Visual Direction

- Black terminal surfaces.
- Violet brand/action states.
- Steel-blue technical accents for manifests, tools, and model state.
- Green reserved for savings, successful execution, and efficiency signals.
- Tight monospace typography.
- No decorative product metaphors. The interface should feel like a serious terminal tool.

See [`style-guide.md`](./style-guide.md) for concrete token usage across docs, assets, and the CLI.
