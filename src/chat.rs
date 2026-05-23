//! Interactive chat REPL.
//!
//! Drives a turn-by-turn conversation with the LLM: read a line from the user
//! via [`rustyline`], stream the model's reply to stdout, push both into the
//! history vector, repeat. Slash-prefixed lines (`/help`, `/clear`, etc.) are
//! intercepted before they reach the model.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use futures_util::StreamExt;
use rustyline::error::ReadlineError;

use crate::provider::{LLMProvider, Message};

/// Run the interactive chat loop until the user exits with `/quit` or Ctrl-D.
///
/// `system`, if provided, is pushed as the first message and preserved across
/// `/clear`. Line-edit history persists to `~/.local/share/yargent/history`.
pub async fn run_chat(
    provider: Box<dyn LLMProvider>,
    system: Option<String>,
) -> Result<()> {
    let mut history: Vec<Message> = Vec::new();
    if let Some(sys) = system {
        history.push(Message::system(sys));
    }

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
            if handle_slash(cmd, &mut history) {
                break;
            }
            continue;
        }

        let _ = rl.add_history_entry(&input);
        history.push(Message::user(input));

        match stream_response(&*provider, &history).await {
            Ok(full) => history.push(Message::assistant(full)),
            Err(e) => {
                eprintln!("\nerror: {e:#}");
                history.pop();
            }
        }
    }

    if let Some(ref p) = history_path {
        let _ = rl.save_history(p);
    }
    Ok(())
}

async fn stream_response(provider: &dyn LLMProvider, history: &[Message]) -> Result<String> {
    let mut stream = provider.complete_stream(history).await?;
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

fn handle_slash(cmd: &str, history: &mut Vec<Message>) -> bool {
    let cmd = cmd.trim();
    match cmd {
        "q" | "quit" | "exit" => return true,
        "clear" => {
            history.retain(|m| m.role == "system");
            println!("[history cleared]");
        }
        "history" => {
            for m in history.iter() {
                println!("--- {} ---", m.role);
                println!("{}", m.content);
            }
        }
        "help" | "?" => print_help(),
        _ => println!("unknown command: /{cmd}  (try /help)"),
    }
    false
}

fn print_help() {
    println!("commands:");
    println!("  /help         show this help");
    println!("  /clear        clear chat history (system prompt preserved)");
    println!("  /history      print full conversation");
    println!("  /quit         exit (or Ctrl-D)");
}

fn history_file_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|p| p.join("yargent").join("history"))
}
