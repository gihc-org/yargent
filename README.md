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
| 2 | File context (`/add foo.rs`, `/drop`, token counts) | ⬜ planned |
| 3 | File editing (SEARCH/REPLACE blocks, diff preview) | ⬜ planned |
| 4 | Git integration via `gix` (auto-commit, `/undo`) | ⬜ planned |
| 5 | Tree-sitter repo map (PageRank over symbol graph) | ⬜ planned |
| 6 | Config file, multi-provider polish | ⬜ planned |

## Quickstart

You need a Rust toolchain (`rustup` or `pkg install rust` on Termux) and an
API key for an LLM provider.

```bash
# DeepSeek is cheap and works well as a default
export DEEPSEEK_API_KEY=sk-...

# Build a release binary (~5.8 MB, no runtime dependencies)
cargo build --release

# One-shot prompt
./target/release/yargent "explain Rust lifetimes in 3 lines"

# Interactive chat — drops into a REPL
./target/release/yargent
```

Override the model with `--model backend/model`:

```bash
./target/release/yargent --model openai/gpt-4o-mini "..."
./target/release/yargent --model anthropic/claude-sonnet-4-6 "..."
```

The backend prefix selects which `*_API_KEY` we read:

| Backend prefix | Env var | Endpoint |
|---|---|---|
| `deepseek/` | `DEEPSEEK_API_KEY` | `api.deepseek.com` |
| `openai/` | `OPENAI_API_KEY` | `api.openai.com` |
| `anthropic/` | `ANTHROPIC_API_KEY` | `api.anthropic.com` |

Prepend a system prompt with `--system`:

```bash
./target/release/yargent --system "Reply only in Danish" "what is async?"
```

Inside the chat REPL, type `/help` for the command list.

## Building on Termux (Android)

```bash
pkg install rust git
git clone <this-repo>
cd yargent
cargo build --release
```

Expect a slow first build (~5–15 min on a phone, depending on the device).
Subsequent rebuilds are seconds. The binary is fully self-contained — no
Python, no glibc-only libraries, no openssl. Just bionic libc and the kernel.

## Architecture

```
src/
├── main.rs       CLI parsing and dispatch
├── provider.rs   LLMProvider trait + OpenAI-compatible backend
└── chat.rs       Interactive REPL with slash commands
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

Direct deps: 11. Indirect: ~150 (normal for an async networking app).

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
- No config file yet; everything goes via CLI flags and `*_API_KEY` env vars.

## License

TBD — this is a personal learning project, not yet released.
