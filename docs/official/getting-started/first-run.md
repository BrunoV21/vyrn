# First Run

Create a global model profile:

```bash
mkdir -p ~/.vyrn
```

```toml
# ~/.vyrn/models.toml
[models.llama3]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
api_key = ""
```

Or scaffold a project-local `.vyrn/` directory with starter config files:

```bash
cargo run -- init
```

The command creates `.vyrn/config.toml`, `.vyrn/models.toml`, and
`.vyrn/skills/` if they do not already exist.

Start with model selection:

```bash
cargo run -- --models
```

Use Up/Down and Enter to choose a profile. `--model` is accepted as an alias.
If no model profiles exist yet, vyrn prompts for a profile name, base URL, model
ID, and optional API key, then saves the profile to `~/.vyrn/models.toml`.

If `vyrn` is installed from Cargo later, use:

```bash
vyrn --models
```

Expected session shape:

```text
vyrn small context first
model llama3  context 4096

> inspect this package and suggest the next implementation step
• I will read the package files and summarize the current state...
turn spent: 812 | turn history saved: 3,204 | session history saved: 11,847 | context: 1,024/4,096
```

Use `/exit` to close the session.

To try the full-screen interface while keeping the normal interface as the
default, launch:

```bash
cargo run -- tui
```

Use PageUp/PageDown for transcript scrolling, End to follow new output, and
Ctrl+K for the command palette.

If the model needs a decision during a turn, vyrn may render an `ask_user`
clarification prompt. Choose an option with the keyboard, or use `Other` for a
freeform reply. In plain text mode, type an option number or any text reply.

## Images

Vision-capable model profiles can receive images with the current message.

```text
> describe ./screen.png and ./mockup.jpg
```

In the TTY composer, `Ctrl+V` attaches an image from the clipboard when one is available.
You can attach multiple images in one message. Supported file types are `png`, `jpg`,
`jpeg`, `webp`, and `gif`; they are sent as base64 data URLs.

Useful local commands inside the session:

```text
/help       list commands and controls
/models     switch model profile
/stats      show provider usage, estimates, and savings
/context    show context used and available
/summary    show the current rolling summary
/scratchpad show the latest turn scratchpad
/manifest   print compact machine manifest
/refresh    rescan manifest
/skills     list discovered skill sources and paths
/debug      show trace status and path
/clear      reset session context and clear the terminal
/exit       close the session
```
