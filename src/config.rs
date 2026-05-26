//! Persistent configuration loaded from `$XDG_CONFIG_HOME/yargent/config.toml`.
//!
//! The whole file is optional — yargent works with zero config and reads API
//! keys from the standard `*_API_KEY` env vars. The config exists to let you:
//!
//! - Set a default model so you don't have to type `--model` every time.
//! - Bake API keys into a file rather than your shell rc.
//! - Define **custom providers** (Ollama, OpenRouter, a private proxy) on top
//!   of yargent's three built-ins (deepseek, openai, anthropic), or override
//!   the built-ins' base URLs.
//! - Tune the repo-map token budget or disable it by default.
//!
//! ## Precedence
//!
//! - **Model selector**: CLI `--model` > `default_model` in config > built-in
//!   fallback (`deepseek/deepseek-chat`).
//! - **API keys**: explicit `api_key` in config > env var named by
//!   `api_key_env` in config > env var `$<NAME>_API_KEY`. We let the config
//!   beat the env var so a fresh user with a stale env var still gets the
//!   behavior they wrote down — pass a different key via env or CLI when you
//!   want a one-off override.
//!
//! ## Example
//!
//! ```toml
//! default_model = "deepseek/deepseek-chat"
//!
//! [providers.deepseek]
//! api_key = "sk-..."
//!
//! [providers.ollama]
//! kind = "openai-compat"
//! base_url = "http://localhost:11434/v1"
//!
//! [providers.openrouter]
//! kind = "openai-compat"
//! base_url = "https://openrouter.ai/api/v1"
//! api_key_env = "OPENROUTER_API_KEY"
//!
//! [repomap]
//! enabled = true
//! token_budget = 2048
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// The top-level config file shape. Every field is optional so an empty or
/// missing file is valid.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Model spec used when `--model` is not given on the CLI.
    pub default_model: Option<String>,
    /// Backend-specific settings keyed by the part before the `/` in a model
    /// spec, e.g. the `deepseek` in `deepseek/deepseek-chat`.
    #[serde(default)]
    pub providers: HashMap<String, ProviderPartial>,
    #[serde(default)]
    pub repomap: RepomapConfig,
    #[serde(default)]
    pub conventions: ConventionsConfig,
}

/// User-overridable provider settings. All fields optional because users
/// frequently want to override just one (api_key for an otherwise default
/// provider, base_url for OpenRouter, etc).
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProviderPartial {
    pub kind: Option<ProviderKind>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    /// Anything that speaks the OpenAI Chat Completions API.
    OpenaiCompat,
    /// Anthropic's Messages API.
    Anthropic,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RepomapConfig {
    /// Whether to inject the repo map into outgoing requests by default.
    /// Override at runtime with `--no-map` or `/map-off`.
    pub enabled: Option<bool>,
    /// Soft cap on repo-map size in tokens.
    pub token_budget: Option<usize>,
}

#[derive(Debug, Default, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ConventionsConfig {
    /// Whether to auto-load convention files at startup. Default: true.
    /// Override at runtime with `--no-conventions`.
    pub enabled: Option<bool>,
    /// Convention filenames to look for in the repo root. If unset, yargent
    /// looks for `AGENTS.md`, `CLAUDE.md`, `CONVENTIONS.md`. Set to `[]` to
    /// disable, or to a custom list to override the defaults wholesale.
    pub paths: Option<Vec<String>>,
}

/// Fully-resolved provider settings ready to construct an LLMProvider.
#[derive(Debug, Clone)]
pub struct ResolvedProvider {
    pub kind: ProviderKind,
    pub base_url: String,
    pub api_key: String,
}

impl Config {
    /// Standard path: `$XDG_CONFIG_HOME/yargent/config.toml`, falling back to
    /// `~/.config/yargent/config.toml` on Linux/Termux and the platform
    /// equivalent elsewhere.
    pub fn default_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("yargent").join("config.toml"))
    }

    /// Load config from `path`. A non-existent file is a successful empty
    /// config — that's the "no setup" common case. A file that exists but
    /// fails to parse surfaces the parse error so the user knows their
    /// config is broken rather than silently ignored.
    pub fn load(path: &Path) -> Result<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => {
                return Err(e).with_context(|| format!("reading config {}", path.display()));
            }
        };
        toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))
    }

    /// Resolve a provider by its name (the `deepseek` in `deepseek/...`).
    ///
    /// The resolution merges three sources, in this precedence:
    /// 1. The user's `[providers.<name>]` table from config.
    /// 2. yargent's built-in defaults for `deepseek`, `openai`, `anthropic`.
    /// 3. Environment variables for the api_key.
    ///
    /// Returns an error with actionable text if a required field can't be
    /// resolved (no api_key found, unknown provider).
    pub fn resolve_provider(&self, name: &str) -> Result<ResolvedProvider> {
        let defaults = builtin_default(name);
        let user = self.providers.get(name).cloned().unwrap_or_default();

        let kind = user
            .kind
            .or(defaults.as_ref().and_then(|d| d.kind))
            .with_context(|| {
                format!(
                    "provider '{name}' has no kind: add `kind = \"openai-compat\"` or \
                     `kind = \"anthropic\"` under [providers.{name}] in the config"
                )
            })?;

        let base_url = user
            .base_url
            .or(defaults.as_ref().and_then(|d| d.base_url.clone()))
            .with_context(|| {
                format!(
                    "provider '{name}' has no base_url: add `base_url = \"...\"` under \
                     [providers.{name}] in the config"
                )
            })?;

        // API key resolution: explicit key wins; otherwise read whichever env
        // var the config (or built-in default) named, falling back to the
        // conventional uppercased-name + "_API_KEY".
        let api_key = if let Some(k) = user.api_key {
            k
        } else {
            let env_var = user
                .api_key_env
                .or(defaults.as_ref().and_then(|d| d.api_key_env.clone()))
                .unwrap_or_else(|| format!("{}_API_KEY", name.to_uppercase()));
            std::env::var(&env_var).with_context(|| {
                format!(
                    "no API key for provider '{name}': set ${env_var} or add `api_key = \
                     \"...\"` under [providers.{name}] in the config"
                )
            })?
        };

        Ok(ResolvedProvider {
            kind,
            base_url,
            api_key,
        })
    }
}

/// Hard-coded defaults for the three providers yargent ships with. Returning
/// `Option` lets [`Config::resolve_provider`] tell "unknown user-defined
/// provider" apart from "known but partly user-overridden".
fn builtin_default(name: &str) -> Option<ProviderPartial> {
    match name {
        "deepseek" => Some(ProviderPartial {
            kind: Some(ProviderKind::OpenaiCompat),
            base_url: Some("https://api.deepseek.com".into()),
            api_key_env: Some("DEEPSEEK_API_KEY".into()),
            api_key: None,
        }),
        "openai" => Some(ProviderPartial {
            kind: Some(ProviderKind::OpenaiCompat),
            base_url: Some("https://api.openai.com".into()),
            api_key_env: Some("OPENAI_API_KEY".into()),
            api_key: None,
        }),
        "anthropic" => Some(ProviderPartial {
            kind: Some(ProviderKind::Anthropic),
            base_url: Some("https://api.anthropic.com".into()),
            api_key_env: Some("ANTHROPIC_API_KEY".into()),
            api_key: None,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// `std::env` is process-global state. `cargo test` runs tests in parallel
    /// within a single binary, so two tests both calling `set_var`/`remove_var`
    /// will race and intermittently assert the wrong thing. Every test below
    /// that mutates env first acquires this lock; tests that don't touch env
    /// (pure parsing, user-defined-provider lookups that resolve from config
    /// without falling back to env) skip it and stay parallel.
    ///
    /// Lock poisoning is intentionally swallowed via `unwrap_or_else(|p|
    /// p.into_inner())` — if a previous test panicked while holding the lock,
    /// we still want subsequent tests to run, and they re-establish the env
    /// state they need explicitly before asserting.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn empty_config_is_default() {
        let cfg: Config = toml::from_str("").unwrap();
        assert!(cfg.default_model.is_none());
        assert!(cfg.providers.is_empty());
    }

    #[test]
    fn parses_full_config() {
        let text = "\
default_model = \"ollama/llama3.2\"

[providers.ollama]
kind = \"openai-compat\"
base_url = \"http://localhost:11434/v1\"

[providers.deepseek]
api_key = \"sk-test\"

[repomap]
enabled = false
token_budget = 2048
";
        let cfg: Config = toml::from_str(text).unwrap();
        assert_eq!(cfg.default_model.as_deref(), Some("ollama/llama3.2"));
        assert_eq!(cfg.repomap.enabled, Some(false));
        assert_eq!(cfg.repomap.token_budget, Some(2048));
        let ollama = cfg.providers.get("ollama").unwrap();
        assert_eq!(ollama.kind, Some(ProviderKind::OpenaiCompat));
        assert_eq!(
            ollama.base_url.as_deref(),
            Some("http://localhost:11434/v1")
        );
    }

    #[test]
    fn resolve_overrides_default_api_key_with_config_value() {
        // User config sets api_key for deepseek; should beat env var.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: env mutation is serialized by ENV_LOCK; no other test
        // holding the lock can race with us.
        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY");
        }
        let cfg: Config = toml::from_str("[providers.deepseek]\napi_key = \"explicit\"\n").unwrap();
        let resolved = cfg.resolve_provider("deepseek").unwrap();
        assert_eq!(resolved.api_key, "explicit");
        assert_eq!(resolved.base_url, "https://api.deepseek.com");
    }

    #[test]
    fn resolve_falls_back_to_env_when_config_silent() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: see resolve_overrides_default_api_key_with_config_value.
        unsafe {
            std::env::set_var("DEEPSEEK_API_KEY", "from-env");
        }
        let cfg = Config::default();
        let resolved = cfg.resolve_provider("deepseek").unwrap();
        assert_eq!(resolved.api_key, "from-env");
        // SAFETY: same lock, same scope — must clean up before releasing
        // the guard so the next lock-holder sees a clean env.
        unsafe {
            std::env::remove_var("DEEPSEEK_API_KEY");
        }
    }

    #[test]
    fn resolve_errors_for_unknown_provider() {
        let cfg = Config::default();
        assert!(cfg.resolve_provider("nonexistent").is_err());
    }

    #[test]
    fn resolve_supports_user_defined_provider() {
        let text = "\
[providers.local]
kind = \"openai-compat\"
base_url = \"http://localhost:8080\"
api_key = \"none\"
";
        let cfg: Config = toml::from_str(text).unwrap();
        let resolved = cfg.resolve_provider("local").unwrap();
        assert_eq!(resolved.kind, ProviderKind::OpenaiCompat);
        assert_eq!(resolved.base_url, "http://localhost:8080");
    }
}
