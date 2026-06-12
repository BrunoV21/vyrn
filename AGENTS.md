# Agent Instructions

## Product Scope

vyrn is currently a Rust package only. Keep documentation, release notes, and examples focused on the Rust CLI package and local development workflow.

Do not introduce npm mirrors, Docker workspaces, hosted inference, GUI surfaces, or multi-agent orchestration unless the product scope changes in `docs/prd.md`.

## Documentation

Official documentation lives in `docs/official/` and is built with VitePress.

Brand positioning and story notes live in `docs/branding/`.

When editing docs:

- Keep examples token-conscious and terminal-native.
- Prefer OpenAI-compatible endpoints in examples.
- Treat small-context and local-model usage as the default path.
- Keep command docs in sync with `docs/prd.md`.

## Releases

When the user asks to create and push a new release or release tag, create release notes before tagging or pushing.

Release notes live in `docs/official/releases/` and use one file per Git tag:

```text
docs/official/releases/vX.Y.Z.md
```

The file name must match the Git tag exactly. The GitHub release workflow can extract the release body from the matching docs file, so the content shown in the docs and in GitHub Releases stays in parity.

Use this structure:

````md
---
title: vyrn vX.Y.Z
description: Release notes for vyrn vX.Y.Z.
---

# vyrn vX.Y.Z

<!-- release-notes:start -->

[GitHub release](https://github.com/BrunoV21/vyrn/releases/tag/vX.Y.Z)

### Highlights

- Added ...
- Fixed ...

### Install

```sh
cargo install vyrn --version X.Y.Z
```

<!-- release-notes:end -->
````

Only content between `<!-- release-notes:start -->` and `<!-- release-notes:end -->` is published as the GitHub release body.

Do not manually edit `docs/official/releases/index.md` for each release. It dynamically reads every `v*.md` file in `docs/official/releases/` and renders a scrollable release page.
