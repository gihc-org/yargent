//! yargent — Rust port of aider.ai. See README.md for the bigger picture.
//!
//! This file is the entry point: it parses the CLI, loads the optional config
//! file, constructs the right [`LLMProvider`] for the requested backend, and
//! dispatches to either [`run_one_shot`] (single prompt → stream to stdout →
//! exit) or [`chat::run_chat`] (interactive REPL).

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use futures_util::StreamExt;

mod chat;
mod config;
mod edit;
mod files;
mod git;
mod provider;
mod repomap;

use config::{Config, ProviderKind};
use provider::{AnthropicProvider, LLMProvider, Message, OpenAICompatProvider};

/// The hard-coded default model used when the user gives neither `--model`
/// nor `default_model` in their config. DeepSeek is cheap and well-supported.
const FALLBACK_MODEL: &str = "deepseek/deepseek-chat";

#[derive(Parser)]
#[command(name = "yargent", version, about = "A Rust AI coding agent")]
struct Cli {
    /// One-shot prompt. If omitted, drops into an interactive chat session.
    prompt: Option<String>,

    /// Provider/model selector, e.g. "deepseek/deepseek-chat" or
    /// "ollama/llama3.2". The part before "/" picks the backend; the
    /// backend can be a built-in (deepseek, openai, anthropic) or anything
    /// defined under `[providers.X]` in the config file.
    #[arg(long)]
    model: Option<String>,

    /// Optional system prompt prepended to the conversation.
    #[arg(long)]
    system: Option<String>,

    /// Path to the config file. Defaults to $XDG_CONFIG_HOME/yargent/config.toml.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Skip auto-commit for this session even when inside a git repo.
    #[arg(long)]
    no_commit: bool,

    /// Skip injecting the repo map for this session.
    #[arg(long)]
    no_map: bool,
}

/// Resolve `--model` → config.default_model → FALLBACK_MODEL.
fn resolve_model_spec(cli_model: Option<&str>, cfg: &Config) -> String {
    cli_model
        .map(str::to_string)
        .or_else(|| cfg.default_model.clone())
        .unwrap_or_else(|| FALLBACK_MODEL.to_string())
}

/// Build a provider for `spec` (in `backend/model` form) using `cfg` for
/// backend lookup. Returns the provider plus the model name string for any
/// logging that wants it.
fn build_provider(spec: &str, cfg: &Config) -> Result<Box<dyn LLMProvider>> {
    let (backend, model) = spec
        .split_once('/')
        .with_context(|| format!("--model must be 'backend/model', got '{spec}'"))?;

    let resolved = cfg.resolve_provider(backend)?;
    match resolved.kind {
        ProviderKind::OpenaiCompat => Ok(Box::new(OpenAICompatProvider::new(
            resolved.base_url,
            resolved.api_key,
            model.to_string(),
        ))),
        ProviderKind::Anthropic => Ok(Box::new(AnthropicProvider::new(
            resolved.base_url,
            resolved.api_key,
            model.to_string(),
        ))),
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

    let config_path = cli
        .config
        .clone()
        .or_else(Config::default_path)
        .unwrap_or_else(|| PathBuf::from("yargent.toml"));
    let config = Config::load(&config_path)?;

    let model_spec = resolve_model_spec(cli.model.as_deref(), &config);
    let provider = build_provider(&model_spec, &config)?;

    match cli.prompt {
        Some(p) => run_one_shot(provider, p, cli.system).await,
        None => {
            let opts = chat::SessionOptions {
                no_commit: cli.no_commit,
                no_map: cli.no_map,
                map_token_budget: config.repomap.token_budget,
                map_default_enabled: config.repomap.enabled.unwrap_or(true),
            };
            chat::run_chat(provider, cli.system, opts).await
        }
    }
}
