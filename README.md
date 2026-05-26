# yargent

A small Rust port of [aider.ai](https://aider.chat) — an AI coding agent that
lives in your terminal. Built to run anywhere Rust runs, including
[Termux](https://termux.dev) on Android, by sticking to pure-Rust dependencies.

## Status

Learning project, built in phases. Each phase ends with something usable.

| Phase | What | State |
|---|---|---|
| 0 | One-shot CLI → LLM round-trip | ✅ done |
| 1 | Streaming, chat REPL, slash commands, persistent history | ✅ done |
| 1.5 | Anthropic provider (Claude) alongside OpenAI-compat | ✅ done |
| 2 | File context (`/add foo.rs`, `/drop`, token counts) | ✅ done |
| 3 | File editing (SEARCH/REPLACE blocks, diff preview) | ✅ done |
| 4 | Git integration (auto-commit, `/undo`) | ✅ done |
| 5 | Tree-sitter repo map (PageRank over symbol graph) | ✅ done |
| 6 | Config file, custom providers, session flags | ✅ done |
| 6.5 | Auto-load convention files (AGENTS.md / CLAUDE.md / …) with `@`-reference following | ✅ done |
| 6.6 | Multi-line REPL input (`{` / `}` blocks and backslash continuation) | ✅ done |

## Daily use

Once installed (`cargo install --path .`) and your API key is in
`~/.config/yargent/config.toml`, daily use is just:

```bash
cd /path/to/your/project
yargent
```

That drops you into the chat REPL. Type your question, press Enter. The
model streams a reply. If it proposes file edits, you'll see a colored
diff and a `[y/n/a/q]` prompt — `y` applies one edit, `a` applies all in
the batch, `n` skips, `q` aborts the remaining edits. Accepted edits are
auto-committed to git in one commit per turn. `/undo` rolls back the
most recent yargent commit.

A few keystrokes worth remembering:

```
/help        list every command (in case you forgot)
/add foo.rs  share a file with the model
/files       show what's shared
/map         print the repo overview
/undo        roll back yargent's last auto-commit
Ctrl-D       quit (same as /quit)
```

If `yargent` says "no API key for provider deepseek", your config file
got moved or deleted. Recreate it:

```bash
mkdir -p ~/.config/yargent
cat > ~/.config/yargent/config.toml << 'EOF'
default_model = "deepseek/deepseek-chat"

[providers.deepseek]
api_key = "sk-YOUR-KEY-HERE"
EOF
chmod 600 ~/.config/yargent/config.toml
```

If `yargent` itself is "command not found", reinstall with
`cargo install --path /home/kristian/projects/yargent` (or wherever you
cloned it).

## First-time setup (Quickstart)

You need a Rust toolchain (`rustup` or `pkg install rust` on Termux) and an
API key for an LLM provider.

```bash
# 1. Build and install yargent globally (~5–10 min first time)
cargo install --path .

# 2. Stash your API key in the config file so you don't have to export
#    it every shell. chmod 600 keeps it readable only by you.
mkdir -p ~/.config/yargent
cat > ~/.config/yargent/config.toml << 'EOF'
default_model = "deepseek/deepseek-chat"

[providers.deepseek]
api_key = "sk-YOUR-KEY-HERE"
EOF
chmod 600 ~/.config/yargent/config.toml

# 3. Use it from anywhere
yargent "explain Rust lifetimes in 3 lines"   # one-shot
yargent                                        # interactive chat
```

Override the model with `--model backend/model`:

```bash
yargent --model openai/gpt-4o-mini "..."
yargent --model anthropic/claude-sonnet-4-6 "..."
```

The backend prefix selects which `*_API_KEY` we read by default:

| Backend prefix | Env var | Endpoint |
|---|---|---|
| `deepseek/` | `DEEPSEEK_API_KEY` | `api.deepseek.com` |
| `openai/` | `OPENAI_API_KEY` | `api.openai.com` |
| `anthropic/` | `ANTHROPIC_API_KEY` | `api.anthropic.com` |

You can also define your own backends in the [config file](#configuration) —
Ollama, OpenRouter, lm-studio, and any other OpenAI-compatible endpoint
become first-class once they're listed under `[providers.X]`.

Prepend a system prompt with `--system`:

```bash
yargent --system "Reply only in Danish" "what is async?"
```

Inside the chat REPL, type `/help` for the command list:

```
/add <path> [path...]   share files with the model
/drop <path> [path...]  stop sharing files
/files                  list currently shared files
/tokens                 estimate tokens in next request
/clear                  clear chat history (system + files preserved)
/history                print full conversation
/help                   show this help
/quit                   exit (or Ctrl-D)
```

Added files are re-read from disk every turn, so edits you make in another
editor propagate without an explicit refresh.

### Multi-line input

By default the chat REPL submits when you press Enter. To enter a
multi-line prompt — pasting code, writing a longer instruction — use one
of two triggers:

- **Block mode**: a line containing only `{` opens a block. Following
  Enter presses insert literal newlines into the buffer rather than
  submitting. A line containing only `}` closes the block and submits.
  The braces are passed to the model verbatim — they're usually
  innocuous and sometimes a useful structure hint.
- **Backslash continuation**: end a line with `\` (literal backslash)
  to continue on the next line. Smaller hammer than block mode for
  one-off two-line prompts.

For block mode the brace must be alone on its line (with optional
surrounding whitespace) — pasted code that happens to contain `{` or
`}` won't accidentally trigger continuation.

```
> {
fix the bug in src/main.rs and add a test
that covers the empty-input case.
}
```

## File editing (SEARCH/REPLACE)

When you ask the model to change code, it emits edits in this format:

````text
src/main.rs
<<<<<<< SEARCH
fn old() {}
=======
fn new() {}
>>>>>>> REPLACE
````

yargent parses every block, shows a colorized unified diff for each, and
prompts before touching disk:

```
--- 2 edit(s) suggested ---

[1/2] src/main.rs
--- src/main.rs
+++ src/main.rs
 fn main() {
-    println!("hello");
+    println!("hello, world");
 }
apply? [y]es / [n]o / [a]ll / [q]uit-prompt:
```

- `y` applies this one edit
- `n` skips it
- `a` applies this one and auto-applies every remaining edit in the batch
- `q` skips all remaining edits in the batch

An empty SEARCH section means "create a new file" — yargent refuses to
overwrite existing files this way, so you can't lose work to a mis-formatted
block. If the SEARCH text doesn't appear in the file (or appears more than
once), that edit is reported and skipped while the others still apply.

The format is taught to the model via a built-in system prompt prepended on
every request. Anything you pass with `--system` is appended after it.

## Git integration

When yargent is run inside a git repo, every batch of successful edits is
auto-committed in one commit per chat turn:

```
yargent: <first line of your prompt, trimmed to 60 chars>

Files:
- src/main.rs

Yargent-Edit: yes
```

The `Yargent-Edit: yes` trailer is what makes `/undo` safe: it does a
`git reset --hard HEAD~1` only when the current HEAD carries the trailer.
Any manual commit you make on top of yargent's auto-commits won't be touched
by `/undo` — you'd unwind those with `git reset` yourself.

Only paths yargent actually touched are staged (no `git add -A`), so dirty
files elsewhere in the working tree stay out of the commit. Caveat: if you
had unstaged changes to a file *before* yargent edited it, those changes get
folded into yargent's auto-commit and would be lost on `/undo`. Commit or
stash before letting yargent change a file you've been editing manually.

Outside a git repo (or if `git` isn't on `PATH`) auto-commit silently
disables itself and the rest of yargent works as normal.

## Repo map

When yargent starts it walks the repo (`.gitignore` respected), parses every
source file it recognizes with tree-sitter, and builds a PageRank-ranked map
of the symbols defined in each file. That map is then injected into every
request alongside any `/add`ed file contents, giving the model a compact
overview of what exists in the codebase without having to see every file in
full.

```
Repo map (PageRank-ranked symbols from files not in the chat):
src/provider.rs:
  Message, LLMProvider, OpenAICompatProvider, AnthropicProvider, ...
src/chat.rs:
  run_chat, ChatState, build_outgoing, handle_edits, ...
...
```

Files you have `/add`ed are *excluded* from the map (the model sees their
full contents elsewhere) but their presence biases the PageRank
personalization vector toward neighbouring files — so the map foregrounds
what's related to your current focus.

Languages currently parsed: **Rust, Python, JavaScript, TypeScript**. Adding
another language is three lines in `src/repomap.rs::Language` plus a query
pair. Files in unrecognized languages are silently skipped.

The rendered map is capped at ~1024 tokens; lowest-ranked files fall off the
end first.

Slash commands:

- `/map` — print the current rendered map
- `/map-on` / `/map-off` — toggle injection on each request
- `/map-rebuild` — re-walk and re-parse the repo

### Why subprocess, not `gix`?

We shell out to the system `git` binary rather than linking a Rust git library:

- `git2` would pull in libgit2 (C) — exactly the kind of native dep we want
  to avoid on Termux/Android.
- `gix` is pure Rust but its write-side API (commit, reset) is still
  maturing and would be substantially more code than its read side.
- `git` is already a prerequisite for any project a user would point
  yargent at, and is one line to install in Termux (`pkg install git`).

Subprocess overhead is invisible at human-latency. Worth revisiting if we
ever need to embed yargent somewhere without a git binary on PATH.

## Convention files

At startup yargent looks in the repo root for any of three "convention
files" — short Markdown documents that describe coding rules or project
context the model should always have in mind. Found files are auto-added
to the chat context exactly as if you had typed `/add` for each.

Defaults:

```
AGENTS.md
CLAUDE.md
CONVENTIONS.md
```

Convention files can either contain the rules inline, or be a list of
`@path` references pulling in other Markdown documents. yargent follows
those references **one level deep** (no recursion — keeps cycle handling
trivial and load behavior predictable). This composes nicely with git
submodules: a one-line `AGENTS.md` like

```markdown
@.guidelines/testing-and-docs.md
```

…pulls in the real conventions from a shared submodule, so multiple
projects can share one set of rules without duplicating them.

Loaded conventions appear in `/files` and can be `/drop`ped like any
other file. Override the auto-loaded filenames in config, or skip the
whole thing for one session with `--no-conventions`:

```toml
[conventions]
enabled = true                            # default true
paths = ["AGENTS.md", "STYLEGUIDE.md"]    # default ["AGENTS.md", "CLAUDE.md", "CONVENTIONS.md"]
```

If an `@`-referenced file is missing the reference is silently skipped,
not an error — the parent convention file still loads, and the bare `@…`
line stays visible to the model so it can react if needed.

## Configuration

yargent runs with zero config — set a `*_API_KEY` env var and you're done.
For anything more, create `$XDG_CONFIG_HOME/yargent/config.toml` (typically
`~/.config/yargent/config.toml`). Every field is optional.

```toml
# Pick this model when --model isn't passed on the CLI.
default_model = "deepseek/deepseek-chat"

# Override a built-in provider — here, bake the API key in.
[providers.deepseek]
api_key = "sk-..."

# Add your own provider. Ollama exposes an OpenAI-compatible endpoint at /v1:
[providers.ollama]
kind = "openai-compat"
base_url = "http://localhost:11434/v1"
api_key = "ollama"        # Ollama doesn't actually check, but the field is required

# OpenRouter — read the key from a custom env var, not OPENROUTER_API_KEY.
[providers.openrouter]
kind = "openai-compat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_KEY"

# Repo-map tuning. Both fields are optional.
[repomap]
enabled = true        # default true; toggle at runtime with /map-on/-off
token_budget = 1024   # default 1024 tokens
```

After defining `[providers.ollama]` you can use `--model ollama/llama3.2`,
and the resolver picks up the base URL and key from the config. The kind
field is one of `"openai-compat"` (most things) or `"anthropic"` (the
Messages API shape — useful if you're proxying Anthropic through your own
service).

**Precedence** (most to least specific):

- API key: explicit `api_key` in config → env var named by `api_key_env` →
  conventional `$<NAME>_API_KEY` env var. Config wins so a stale env var
  doesn't quietly break a working setup.
- Model: `--model` CLI flag → `default_model` in config → built-in
  `deepseek/deepseek-chat`.

### Session flags

A few flags switch features off for one session without editing config:

```bash
yargent --no-commit "..."        # don't auto-commit edits
yargent --no-map "..."           # don't inject the repo map
yargent --no-conventions "..."   # don't auto-load AGENTS.md/CLAUDE.md/CONVENTIONS.md
yargent --config /tmp/test.toml  # use a different config file
```

## Building on Termux (Android)

```bash
pkg install rust git
git clone <this-repo>
cd yargent
cargo build --release
```

Expect a slow first build (~5–15 min on a phone, depending on the device).
Subsequent rebuilds are seconds. The binary links bionic libc and the kernel,
plus the small C parsers tree-sitter compiles from its grammar crates — no
Python, no openssl, no libgit2, nothing that historically broke on Termux.

You'll need `clang` available for the tree-sitter grammars to build:

```bash
pkg install rust git clang
```

## Architecture

```
src/
├── main.rs       CLI parsing and dispatch
├── provider.rs   LLMProvider trait + OpenAI-compatible + Anthropic backends
├── chat.rs       Interactive REPL with slash commands and file context
├── files.rs      FileContext: tracks added paths, renders them for the model
├── edit.rs       SEARCH/REPLACE parser, applier, diff preview, system prompt
├── git.rs        Repo discovery + auto-commit + /undo via subprocess git
├── repomap.rs    Tree-sitter walker + PageRank symbol map injection
└── config.rs     TOML config: default model, custom providers, repo-map tuning
```

### The core trait

```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn complete_stream(&self, messages: &[Message]) -> Result<ChunkStream>;
}
```

One method, one returned stream of text chunks. Every backend is a plug-in.

`OpenAICompatProvider` covers a large ecosystem because **DeepSeek, OpenAI,
Groq, OpenRouter, Together, Mistral, lm-studio and Ollama** all speak the same
`/v1/chat/completions` dialect — they differ only in base URL and API key.

`AnthropicProvider` handles Claude separately because the Messages API has its
own shape: `system` is a top-level field rather than a `role: "system"` message,
`max_tokens` is required, auth uses `x-api-key` instead of `Bearer`, and the
streaming events are tagged with `type` rather than nested under `choices[].delta`.
The trait hides all of this from callers.

### Backend selection

The `--model` flag is `backend/model`:

```
--model deepseek/deepseek-chat
        └─backend └─model name (passed verbatim)
```

`main.rs::build_provider` matches the backend prefix and constructs the right
`Box<dyn LLMProvider>` for the runtime dispatch.

### Why these dependencies?

We pick pure-Rust crates over C-bindings wherever possible. This matters for
Termux/Android where bionic libc and missing system libraries make C-based
builds painful, and it also keeps cross-compilation simple.

| Crate | Used for | Why this one |
|---|---|---|
| `tokio` | async runtime | Standard. Multi-threaded so `block_in_place` works around blocking calls. |
| `reqwest` | HTTP client | `rustls-tls` feature → pure-Rust TLS, no openssl. `stream` feature → SSE chunks. |
| `serde` + `serde_json` | JSON | Standard. |
| `clap` | CLI parsing | Standard. `derive` lets us declare args as struct fields. |
| `anyhow` | error handling | `?`-propagation with `.context()`. Right for a binary; libraries should use `thiserror`. |
| `async-trait` | async fn in trait | Native async-fn-in-trait doesn't yet support `dyn Trait`. |
| `futures-util` | `Stream`, `StreamExt` | The async equivalent of `Iterator`. |
| `async-stream` | `try_stream!` macro | Write streams in straight-line code with `yield`. |
| `rustyline` | REPL line editing | Arrow keys, history file, Ctrl-R search. |
| `dirs` | XDG paths | Cross-platform `~/.local/share/...` resolution. |
| `tiktoken-rs` | Token counting | Pure-Rust BPE, bundles its own merge tables. `cl100k_base` is exact for OpenAI and a close estimate for DeepSeek/Anthropic. |
| `similar` | Diff rendering | Pure-Rust. We use `TextDiff::from_lines` for the unified-diff preview shown before applying edits. ANSI colors are bare escape sequences, no extra terminal crate. |
| `tree-sitter` + grammars | Repo-map parsing | C-based but small and notoriously portable. Grammar crates compile their own C parser via `build.rs`; on Termux this needs `pkg install clang`. |
| `ignore` | File walker | From the ripgrep ecosystem. Pure-Rust. Respects `.gitignore` so the map doesn't drown in `target/` and `node_modules/`. |
| `toml` | Config parser | Pure-Rust TOML for `$XDG_CONFIG_HOME/yargent/config.toml`. |

Direct deps: 20. Indirect: ~175 (normal for an async networking app with parsers).

## Rust concepts in this codebase

If you're reading the code to learn Rust, here's a treasure map:

- **Traits + dynamic dispatch** — `Box<dyn LLMProvider>` in `main.rs`.
  The compiler builds a vtable at runtime so we can swap providers without
  changing call sites. Static dispatch (generics) would also work but couldn't
  be selected by a runtime string.
- **`Result<T, E>` and `?`** — every fallible function returns `Result`. The
  `?` operator early-returns on `Err`. `anyhow::Result<T>` is shorthand for
  `Result<T, anyhow::Error>`.
- **`.context("...")`** — attaches a human-readable layer to an error chain
  without losing the underlying cause. Errors print outer-context first,
  root cause last.
- **`async fn` + `.await`** — sequential async. You write code that looks
  synchronous; tokio handles the scheduling.
- **`Stream<Item = T>`** — async iterator. See `provider.rs::complete_stream`
  for how to build one with `async_stream::try_stream!`, and
  `chat.rs::stream_response` for how to consume one with `.next().await`.
- **`Pin<Box<dyn Stream + Send>>`** — required wrapper around type-erased
  streams. `Pin` guarantees the value won't be moved (necessary because
  async state machines can be self-referential).
- **`impl Into<String>`** — lets `Message::user("hi")` and
  `Message::user(String::from("hi"))` both work. Many types implement
  `Into<String>`.
- **`tokio::task::block_in_place`** — see `chat.rs`. Tells tokio "I'm about
  to block this thread on something outside the async runtime; go schedule
  other tasks on different worker threads."
- **Module system** — `mod provider;` in `main.rs` pulls in `src/provider.rs`.
  Items marked `pub` are visible across module boundaries.

## Development

```bash
cargo check                # Fast type-check, no codegen
cargo build                # Debug build (fast compile, slow run)
cargo build --release      # Optimized
cargo clippy               # Lints
cargo fmt                  # Auto-format
cargo doc --open           # Browse generated API docs in a browser
cargo run -- "test prompt" # Build + run debug build
```

### Files & data

- History file: `~/.local/share/yargent/history` (Linux/Termux — XDG data dir).
- Config file: `~/.config/yargent/config.toml` (optional). See
  [Configuration](#configuration) above.

## License

TBD — this is a personal learning project, not yet released.
