//! yargent — Rust port of aider.ai. See README.md for the bigger picture.
//!
//! This file is the entry point: it parses the CLI, constructs the right
//! [`LLMProvider`] for the requested backend, and dispatches to either
//! [`run_one_shot`] (single prompt → stream to stdout → exit) or
//! [`chat::run_chat`] (interactive REPL).

use std::io::Write;

use anyhow::{Context, Result};
use clap::Parser;
use futures_util::StreamExt;

mod chat;
mod edit;
mod files;
mod git;
mod provider;
mod repomap;
use provider::{AnthropicProvider, LLMProvider, Message, OpenAICompatProvider};

#[derive(Parser)]
#[command(name = "yargent", version, about = "A Rust AI coding agent")]
struct Cli {
    /// One-shot prompt. If omitted, drops into an interactive chat session.
    prompt: Option<String>,

    /// Provider/model selector, e.g. "deepseek/deepseek-chat" or
    /// "openai/gpt-4o-mini". The part before "/" picks the backend.
    #[arg(long, default_value = "deepseek/deepseek-chat")]
    model: String,

    /// Optional system prompt prepended to the conversation.
    #[arg(long)]
    system: Option<String>,
}

fn build_provider(spec: &str) -> Result<Box<dyn LLMProvider>> {
    let (backend, model) = spec
        .split_once('/')
        .with_context(|| format!("--model must be 'backend/model', got '{spec}'"))?;

    match backend {
        "deepseek" => {
            let key = std::env::var("DEEPSEEK_API_KEY")
                .context("DEEPSEEK_API_KEY environment variable is not set")?;
            Ok(Box::new(OpenAICompatProvider::deepseek(key, model.to_string())))
        }
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY")
                .context("OPENAI_API_KEY environment variable is not set")?;
            Ok(Box::new(OpenAICompatProvider::openai(key, model.to_string())))
        }
        "anthropic" => {
            let key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY environment variable is not set")?;
            Ok(Box::new(AnthropicProvider::new(key, model.to_string())))
        }
        other => anyhow::bail!(
            "unknown backend '{other}'. Supported: deepseek, openai, anthropic"
        ),
    }
}

async fn run_one_shot(
    provider: Box<dyn LLMProvider>,
    prompt: String,
    system: Option<String>,
) -> Result<()> {
    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(Message::system(sys));
    }
    messages.push(Message::user(prompt));

    let mut stream = provider.complete_stream(&messages).await?;
    let mut stdout = std::io::stdout();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{chunk}");
        stdout.flush().ok();
    }
    println!();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let provider = build_provider(&cli.model)?;

    match cli.prompt {
        Some(p) => run_one_shot(provider, p, cli.system).await,
        None => chat::run_chat(provider, cli.system).await,
    }
}
