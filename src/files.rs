//! File context: the set of files the user has added to the chat with
//! [`/add`](FileContext::add).
//!
//! We track *paths*, not contents. Each time we send a message to the model we
//! re-read every added file from disk via [`render`](FileContext::render). That
//! way edits the user makes between turns (in another editor, or that the model
//! itself applied in a future phase) propagate automatically without an explicit
//! `/refresh` command.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Ordered set of file paths currently shared with the model.
///
/// Paths are stored canonicalized so `./src/main.rs` and `src/main.rs` resolve
/// to the same entry. We iterate in sorted order (that's what `BTreeSet` gives
/// us) so the rendered block is stable turn-to-turn — important for prompt
/// caching on providers that key on prefix.
#[derive(Default)]
pub struct FileContext {
    paths: BTreeSet<PathBuf>,
}

impl FileContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file by path. Errors if the path does not point at a readable
    /// regular file. Adding the same path twice is a no-op.
    pub fn add(&mut self, path: impl AsRef<Path>) -> Result<PathBuf> {
        let raw = path.as_ref();
        let canon = raw
            .canonicalize()
            .with_context(|| format!("cannot resolve path '{}'", raw.display()))?;
        if !canon.is_file() {
            anyhow::bail!("'{}' is not a regular file", raw.display());
        }
        self.paths.insert(canon.clone());
        Ok(canon)
    }

    /// Remove a file by path. Accepts either the canonical or any equivalent
    /// path the user originally typed. Returns true if a file was actually
    /// removed.
    pub fn drop(&mut self, path: impl AsRef<Path>) -> bool {
        let raw = path.as_ref();
        if let Ok(canon) = raw.canonicalize()
            && self.paths.remove(&canon)
        {
            return true;
        }
        self.paths.remove(raw)
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    pub fn paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.paths.iter()
    }

    /// Render every added file as a single Markdown-ish text block ready to be
    /// sent to the model. Format:
    ///
    /// ```text
    /// I have added these files to the chat. Their full contents follow:
    ///
    /// src/main.rs:
    /// ```rust
    /// <contents>
    /// ```
    ///
    /// src/lib.rs:
    /// ```rust
    /// <contents>
    /// ```
    /// ```
    ///
    /// Paths are shown relative to the current working directory when possible,
    /// so the model sees `src/main.rs` rather than the full canonical path.
    pub fn render(&self) -> Result<String> {
        let mut out =
            String::from("I have added these files to the chat. Their full contents follow:\n");
        for path in &self.paths {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let display = display_path(path);
            let lang = lang_for_path(path);
            out.push('\n');
            out.push_str(&display);
            out.push_str(":\n```");
            out.push_str(lang);
            out.push('\n');
            out.push_str(&content);
            if !content.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("```\n");
        }
        Ok(out)
    }
}

/// Render `path` relative to the current working directory if it's underneath
/// it, otherwise fall back to the absolute path.
fn display_path(path: &Path) -> String {
    if let Ok(cwd) = std::env::current_dir()
        && let Ok(rel) = path.strip_prefix(&cwd)
    {
        return rel.display().to_string();
    }
    path.display().to_string()
}

/// Map a file extension to a Markdown info-string used in fenced code blocks.
/// Improves syntax-highlighting in model UIs and gives the model a clearer hint
/// about what language it's looking at.
fn lang_for_path(path: &Path) -> &'static str {
    let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
        return "";
    };
    match ext {
        "rs" => "rust",
        "py" => "python",
        "js" | "mjs" | "cjs" => "javascript",
        "ts" | "tsx" => "typescript",
        "go" => "go",
        "c" | "h" => "c",
        "cpp" | "cc" | "cxx" | "hpp" | "hh" => "cpp",
        "java" => "java",
        "kt" | "kts" => "kotlin",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "md" => "markdown",
        "toml" => "toml",
        "yml" | "yaml" => "yaml",
        "json" => "json",
        "html" | "htm" => "html",
        "css" => "css",
        "sh" | "bash" => "bash",
        "sql" => "sql",
        "lua" => "lua",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn render_emits_file_with_language_fence() {
        let dir = tempdir();
        let path = dir.join("hello.rs");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "fn main() {{}}").unwrap();

        let mut ctx = FileContext::new();
        ctx.add(&path).unwrap();
        let rendered = ctx.render().unwrap();

        assert!(rendered.contains("```rust"));
        assert!(rendered.contains("fn main() {}"));
    }

    #[test]
    fn drop_removes_added_file() {
        let dir = tempdir();
        let path = dir.join("a.txt");
        std::fs::File::create(&path).unwrap();

        let mut ctx = FileContext::new();
        ctx.add(&path).unwrap();
        assert!(!ctx.is_empty());
        assert!(ctx.drop(&path));
        assert!(ctx.is_empty());
        assert!(!ctx.drop(&path), "second drop should report no-op");
    }

    /// Create a unique temp directory under the OS tempdir. We roll our own so
    /// we don't need a `tempfile` dependency just for tests.
    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("yargent-test-{pid}-{n}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
