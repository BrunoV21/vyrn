---
layout: home

hero:
  name: "vyrn"
  text: "Token-efficient agents for small contexts"
  tagline: "A Rust CLI agent for local and small OpenAI-compatible models."
  actions:
    - theme: brand
      text: "Get Started"
      link: /getting-started/
    - theme: alt
      text: "Commands"
      link: /commands/
    - theme: alt
      text: "Architecture"
      link: /concepts/architecture
    - theme: alt
      text: "GitHub"
      link: https://github.com/BrunoV21/vyrn

features:
  - title: "Small context first"
    details: "The prompt, machine manifest, core tools, and history model are built for limited context windows."
  - title: "Rust package"
    details: "vyrn is scoped as a Rust CLI package. No npm mirror, hosted service, Docker workspace, or GUI is part of the current product."
  - title: "OpenAI compatible"
    details: "Connect to Ollama, LM Studio, Groq, Together AI, OpenRouter, or any endpoint that implements OpenAI chat completions."
  - title: "Rolling summaries"
    details: "Each request refreshes a compact summary instead of resending full conversation history."
  - title: "Raw batch power"
    details: "The batch tool is the primary extension primitive for shell work, scripts, installs, and host inspection."
  - title: "Token savings"
    details: "Every completed turn reports tokens spent, tokens saved, and total saved for the session."
  - title: "Open standards"
    details: "Skills use Agent Skills protocol, while MCP configuration follows .mcp.json conventions."
---

<div class="vy-terminal">
<strong>$ vyrn --models</strong><br>
<span class="grid">vyrn small context first</span><br>
model llama3 context 4096<br>
<br>
&gt; summarize this repo and find the next implementation step<br>
<span class="ok">turn spent: 812 | turn saved: 3,204 | session saved: 11,847 | context: 1,024/4,096</span>
</div>

## The idea

Most CLI agents are comfortable spending a large context budget before the user asks for anything. vyrn is built for the opposite environment: small local models, fast terminal workflows, and users who care about every token that enters the prompt.

vyrn keeps the core prompt and tool list tiny, makes capability load progressively through skills and MCP discovery, and treats token savings as visible product behavior.

## Install

```bash
git clone https://github.com/BrunoV21/vyrn.git
cd vyrn
cargo build
cargo test
```

Once published:

```bash
cargo install vyrn
vyrn --models
```

## Next steps

1. Read the [getting started guide](./getting-started/).
2. Review the [command surface](./commands/).
3. Understand [rolling context management](./concepts/context-management.md).
4. Check the [roadmap](./roadmap.md).
