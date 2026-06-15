# vyrn Style Guide

vyrn should look like a constrained-compute tool, not a generic neon AI product.

## Palette

### Core neutrals

- `--vy-bg: #06070A`
- `--vy-surface: #0D1016`
- `--vy-surface-raised: #151A22`
- `--vy-border: #273142`
- `--vy-border-strong: #3A475E`
- `--vy-text-primary: #F3F7FB`
- `--vy-text-muted: #98A3B3`
- `--vy-text-dim: #677287`

### Brand and semantic colors

- `--vy-violet: #8B5CF6`
- `--vy-violet-hover: #A78BFA`
- `--vy-violet-active: #7C3AED`
- `--vy-tech: #7DA2C2`
- `--vy-tech-strong: #A9BDD3`
- `--vy-success: #9FE870`
- `--vy-amber: #F5A524`
- `--vy-red: #F43F5E`

## Roles

- `violet`: identity, selected states, primary actions
- `steel blue`: model info, manifests, tools, system framing, technical links
- `green`: token savings, confirmations, healthy completion
- `amber`: warnings, high context pressure
- `red`: failures, rejected actions, provider errors

## Docs rules

- Keep backgrounds neutral and low-gloss.
- Use steel blue for technical emphasis, not as a second brand color.
- Keep green out of decorative gradients and general navigation.
- Feature cards should feel like terminal panels, not marketing tiles.
- Inline code, system labels, and raw-doc links should read as operator tooling.

## CLI rules

- Welcome banner and model picker use steel blue rather than cyan.
- Prompt framing and technical status elements use dark graphite and steel-blue accents.
- Savings remain green.
- Errors remain red.
- Avoid rainbow banners or mixed accent stacks unless a value has semantic meaning.

## Asset rules

- Banner art should use violet for the product mark, steel blue for technical UI cues, and green only for savings.
- Favicon and ASCII surfaces should stay minimal and avoid decorative gradients where possible.
