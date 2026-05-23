//! Interactive chat REPL.
//!
//! Drives a turn-by-turn conversation with the LLM: read a line from the user
//! via [`rustyline`], stream the model's reply to stdout, push both into the
//! history vector, repeat. Slash-prefixed lines (`/help`, `/clear`, `/add`,
//! etc.) are intercepted before they reach the model.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use futures_util::StreamExt;
use rustyline::error::ReadlineError;
use tiktoken_rs::CoreBPE;

use crate::files::FileContext;
use crate::provider::{LLMProvider, Message};

/// Run the interactive chat loop until the user exits with `/quit` or Ctrl-D.
///
/// `system`, if provided, is pushed as the first message and preserved across
/// `/clear`. Line-edit history persists to `~/.local/share/yargent/history`.
pub async fn run_chat(provider: Box<dyn LLMProvider>, system: Option<String>) -> Result<()> {
    let tokenizer = tiktoken_rs::cl100k_base()?;
    let mut state = ChatState::new(system, tokenizer);

    let mut rl = rustyline::DefaultEditor::new()?;
    let history_path = history_file_path();
    if let Some(ref p) = history_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = rl.load_history(p);
    }

    println!("yargent — type /help for commands, Ctrl-D to quit");

    loop {
        // rustyline's readline is synchronous and blocks the thread waiting
        // for input. `block_in_place` tells tokio's multi-threaded runtime to
        // move other tasks off this worker thread while we block here, so we
        // don't stall the rest of the runtime.
        let line = tokio::task::block_in_place(|| rl.readline("> "));
        let input = match line {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(cmd) = trimmed.strip_prefix('/') {
            if handle_slash(cmd, &mut state) {
                break;
            }
            continue;
        }

        let _ = rl.add_history_entry(&input);
        state.history.push(Message::user(input));

        let outgoing = match state.build_outgoing() {
            Ok(msgs) => msgs,
            Err(e) => {
                eprintln!("error preparing message: {e:#}");
                state.history.pop();
                continue;
            }
        };

        match stream_response(&*provider, &outgoing).await {
            Ok(full) => state.history.push(Message::assistant(full)),
            Err(e) => {
                eprintln!("\nerror: {e:#}");
                state.history.pop();
            }
        }
    }

    if let Some(ref p) = history_path {
        let _ = rl.save_history(p);
    }
    Ok(())
}

/// Everything we mutate across the chat loop in one bundle.
///
/// Pulled out into a struct so `handle_slash` can borrow it mutably as a whole
/// rather than juggling N parallel `&mut` borrows.
struct ChatState {
    /// User/assistant/system turns as the user sees them — clean, never
    /// polluted by injected file content or other synthetic messages.
    history: Vec<Message>,
    /// Files the user has shared with `/add`. Re-read on every turn.
    files: FileContext,
    /// BPE tokenizer used for `/tokens` estimates. Loaded once; the merge
    /// tables are bundled into the binary so this never touches the network.
    tokenizer: CoreBPE,
}

impl ChatState {
    fn new(system: Option<String>, tokenizer: CoreBPE) -> Self {
        let mut history = Vec::new();
        if let Some(sys) = system {
            history.push(Message::system(sys));
        }
        Self {
            history,
            files: FileContext::new(),
            tokenizer,
        }
    }

    /// Build the actual list of messages to send to the provider this turn.
    ///
    /// If any files have been `/add`ed, splice in a synthetic user message
    /// containing their current contents plus a one-line assistant
    /// acknowledgement, placed *after* the system prompt but *before* the
    /// rest of the conversation. The injection is regenerated every turn so
    /// edits propagate, and it never appears in `self.history` — that stays
    /// the clean record of what the user actually typed.
    fn build_outgoing(&self) -> Result<Vec<Message>> {
        if self.files.is_empty() {
            return Ok(self.history.clone());
        }

        let rendered = self.files.render()?;
        let mut out = Vec::with_capacity(self.history.len() + 2);

        // System messages (if any) stay at the very front. By convention there
        // is at most one, but we handle multiple defensively.
        let split_at = self
            .history
            .iter()
            .position(|m| m.role != "system")
            .unwrap_or(self.history.len());

        out.extend_from_slice(&self.history[..split_at]);
        out.push(Message::user(rendered));
        out.push(Message::assistant(
            "Got it. I'll work with those files.",
        ));
        out.extend_from_slice(&self.history[split_at..]);
        Ok(out)
    }

    /// Rough token-count estimate of what we'd send next turn. Uses
    /// `cl100k_base` — exact for OpenAI models, an approximation for others.
    fn token_estimate(&self) -> usize {
        let outgoing = match self.build_outgoing() {
            Ok(msgs) => msgs,
            Err(_) => return 0,
        };
        outgoing
            .iter()
            .map(|m| self.tokenizer.encode_with_special_tokens(&m.content).len())
            .sum()
    }
}

async fn stream_response(provider: &dyn LLMProvider, messages: &[Message]) -> Result<String> {
    let mut stream = provider.complete_stream(messages).await?;
    let mut full = String::new();
    let mut stdout = std::io::stdout();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{chunk}");
        stdout.flush().ok();
        full.push_str(&chunk);
    }
    println!();
    Ok(full)
}

/// Dispatch a slash command. Returns true if the loop should exit.
fn handle_slash(cmd: &str, state: &mut ChatState) -> bool {
    let cmd = cmd.trim();

    // Commands that take arguments — split off the head word.
    let (head, rest) = match cmd.split_once(char::is_whitespace) {
        Some((h, r)) => (h, r.trim()),
        None => (cmd, ""),
    };

    match head {
        "q" | "quit" | "exit" => return true,
        "clear" => {
            state.history.retain(|m| m.role == "system");
            println!("[history cleared]");
        }
        "history" => {
            for m in &state.history {
                println!("--- {} ---", m.role);
                println!("{}", m.content);
            }
        }
        "add" => {
            if rest.is_empty() {
                println!("usage: /add <path> [path...]");
            } else {
                for path in rest.split_whitespace() {
                    match state.files.add(path) {
                        Ok(canon) => println!("[added {}]", canon.display()),
                        Err(e) => println!("error: {e:#}"),
                    }
                }
            }
        }
        "drop" => {
            if rest.is_empty() {
                println!("usage: /drop <path> [path...]");
            } else {
                for path in rest.split_whitespace() {
                    if state.files.drop(path) {
                        println!("[dropped {path}]");
                    } else {
                        println!("[not in context: {path}]");
                    }
                }
            }
        }
        "files" | "ls" => {
            if state.files.is_empty() {
                println!("[no files added]");
            } else {
                for p in state.files.paths() {
                    println!("  {}", p.display());
                }
            }
        }
        "tokens" => {
            let n = state.token_estimate();
            println!("[~{n} tokens in next request (cl100k_base estimate)]");
        }
        "help" | "?" => print_help(),
        _ => println!("unknown command: /{cmd}  (try /help)"),
    }
    false
}

fn print_help() {
    println!("commands:");
    println!("  /add <path> [path...]   share files with the model");
    println!("  /drop <path> [path...]  stop sharing files");
    println!("  /files                  list currently shared files");
    println!("  /tokens                 estimate tokens in next request");
    println!("  /clear                  clear chat history (system + files preserved)");
    println!("  /history                print full conversation");
    println!("  /help                   show this help");
    println!("  /quit                   exit (or Ctrl-D)");
}

fn history_file_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|p| p.join("yargent").join("history"))
}
