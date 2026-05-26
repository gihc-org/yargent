//! File context: the set of files the user has added to the chat with
//! [`/add`](FileContext::add).
//!
//! We track *paths*, not contents. Each time we send a message to the model we
//! re-read every added file from disk via [`render`](FileContext::render). That
//! way edits the user makes between turns (in another editor, or that the model
//! itself applied in a future phase) propagate automatically without an explicit
//! `/refresh` command.

use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Filenames yargent auto-loads from the repo root as "convention files" —
/// short Markdown documents that describe coding rules, conventions, or
/// general project context. Anything found here is added to the file context
/// at startup as if the user had typed `/add` for it.
///
/// Drop one of these in your repo root and yargent picks it up next launch.
/// Override the list per project via `[conventions] paths = [...]` in the
/// config file.
pub const DEFAULT_CONVENTION_NAMES: &[&str] = &["AGENTS.md", "CLAUDE.md", "CONVENTIONS.md"];

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

    /// Add a path. If it's a regular file, add it directly. If it's a
    /// directory, walk recursively (respecting `.gitignore` via
    /// `ignore::WalkBuilder`) and add every regular file found inside.
    /// Returns the canonical path of the *directory* when a directory is
    /// expanded, so the caller (e.g. `/add` output) shows the original
    /// name. Adding the same path twice is a no-op.
    pub fn add(&mut self, path: impl AsRef<Path>) -> Result<PathBuf> {
        let raw = path.as_ref();
        let canon = raw
            .canonicalize()
            .with_context(|| format!("cannot resolve path '{}'", raw.display()))?;
        if canon.is_dir() {
            let walker = ignore::WalkBuilder::new(&canon).build();
            for entry in walker.flatten() {
                let p = entry.path();
                if p.is_file() {
                    match p.canonicalize() {
                        Ok(c) => {
                            self.paths.insert(c);
                        }
                        Err(e) => {
                            eprintln!(
                                "warning: cannot canonicalize '{}' — {}",
                                p.display(),
                                e
                            );
                        }
                    }
                }
            }
            Ok(canon)
        } else if canon.is_file() {
            self.add_single(canon)
        } else {
            anyhow::bail!("'{}' is not a regular file or directory", raw.display());
        }
    }

    /// Add a single, already-canonicalised regular file. Errors if
    /// `canon` doesn't point at a readable regular file.
    fn add_single(&mut self, canon: PathBuf) -> Result<PathBuf> {
        if !canon.is_file() {
            anyhow::bail!("'{}' is not a regular file", canon.display());
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

    pub fn len(&self) -> usize {
        self.paths.len()
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

/// Find convention files under `root` and any single-level `@path` references
/// they contain, returning canonical absolute paths suitable for
/// [`FileContext::add`].
///
/// A "convention file" is a top-level Markdown file matching one of `names`
/// — typically [`DEFAULT_CONVENTION_NAMES`]. We additionally follow the
/// aider/Claude Code convention where the file is allowed to be just a list
/// of `@relative/path` lines, each pulling in another file. We follow those
/// references *one level deep* — references inside referenced files are
/// silently ignored to keep cycles impossible and the loading deterministic.
///
/// Missing convention files are silently skipped (the common case is that
/// most projects don't have any). Missing `@`-referenced files are also
/// silently skipped — the referencing convention file is still loaded, so
/// the model at least sees the unresolved reference and can react to it.
pub fn discover_conventions(root: &Path, names: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    for name in names {
        let path = root.join(name);
        if !path.is_file() {
            continue;
        }
        let Ok(canon) = path.canonicalize() else { continue };
        if !seen.insert(canon.clone()) {
            continue;
        }
        out.push(canon.clone());

        // Single-pass: scan this convention file for @path references and
        // pull in anything that resolves. Recursive expansion is deliberately
        // not implemented to avoid cycles and to keep load behavior obvious.
        let Ok(content) = std::fs::read_to_string(&canon) else { continue };
        let parent = canon.parent().unwrap_or_else(|| Path::new("."));
        for line in content.lines() {
            let Some(rest) = line.trim().strip_prefix('@') else {
                continue;
            };
            let ref_path = parent.join(rest.trim());
            if !ref_path.is_file() {
                continue;
            }
            if let Ok(ref_canon) = ref_path.canonicalize()
                && seen.insert(ref_canon.clone())
            {
                out.push(ref_canon);
            }
        }
    }

    out
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
    fn discover_returns_empty_when_no_convention_files() {
        let dir = tempdir();
        let found = discover_conventions(&dir, DEFAULT_CONVENTION_NAMES);
        assert!(found.is_empty());
    }

    #[test]
    fn discover_finds_plain_convention_file() {
        let dir = tempdir();
        std::fs::write(dir.join("AGENTS.md"), "be tidy.\n").unwrap();
        let found = discover_conventions(&dir, DEFAULT_CONVENTION_NAMES);
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("AGENTS.md"));
    }

    #[test]
    fn discover_follows_at_references_one_level() {
        let dir = tempdir();
        // AGENTS.md contains a @reference to another file in the same dir.
        std::fs::write(dir.join("AGENTS.md"), "@style.md\n").unwrap();
        std::fs::write(dir.join("style.md"), "use four spaces.\n").unwrap();

        let found = discover_conventions(&dir, DEFAULT_CONVENTION_NAMES);
        assert_eq!(found.len(), 2, "expected AGENTS.md + style.md, got {found:?}");
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"AGENTS.md".to_string()));
        assert!(names.contains(&"style.md".to_string()));
    }

    #[test]
    fn discover_ignores_missing_at_reference() {
        let dir = tempdir();
        std::fs::write(dir.join("AGENTS.md"), "@nonexistent.md\nstill here\n").unwrap();
        let found = discover_conventions(&dir, DEFAULT_CONVENTION_NAMES);
        assert_eq!(found.len(), 1, "missing references should not block loading");
        assert!(found[0].ends_with("AGENTS.md"));
    }

    #[test]
    fn discover_does_not_recurse_into_referenced_file() {
        // a.md → @b.md; b.md → @c.md. We should pick up a and b but NOT c.
        let dir = tempdir();
        std::fs::write(dir.join("AGENTS.md"), "@b.md\n").unwrap();
        std::fs::write(dir.join("b.md"), "@c.md\n").unwrap();
        std::fs::write(dir.join("c.md"), "should not be loaded\n").unwrap();

        let found = discover_conventions(&dir, DEFAULT_CONVENTION_NAMES);
        let names: Vec<_> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"AGENTS.md".to_string()));
        assert!(names.contains(&"b.md".to_string()));
        assert!(
            !names.contains(&"c.md".to_string()),
            "depth > 1 should not be followed (got {names:?})"
        );
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

    #[test]
    fn add_empty_directory_adds_no_files() {
        // Empty dir is accepted (no error) but produces zero new files —
        // the count display in /add uses this to print "(0 files)" rather
        // than silently lying that something was added.
        let dir = tempdir();
        let empty = dir.join("empty");
        std::fs::create_dir(&empty).unwrap();

        let mut ctx = FileContext::new();
        let before = ctx.len();
        let canon = ctx.add(&empty).unwrap();

        assert_eq!(ctx.len(), before, "no files expected from an empty dir");
        assert!(canon.is_dir(), "should return the dir's canonical path");
    }

    #[test]
    fn re_adding_same_file_is_idempotent() {
        let dir = tempdir();
        let f = dir.join("only.rs");
        std::fs::write(&f, "fn main() {}\n").unwrap();

        let mut ctx = FileContext::new();
        ctx.add(&f).unwrap();
        let after_first = ctx.len();
        ctx.add(&f).unwrap();

        assert_eq!(
            ctx.len(),
            after_first,
            "re-adding the same file should leave the set unchanged"
        );
    }

    #[test]
    fn re_adding_same_directory_adds_nothing_new() {
        let dir = tempdir();
        let sub = dir.join("project");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("a.rs"), "// a\n").unwrap();
        std::fs::write(sub.join("b.rs"), "// b\n").unwrap();

        let mut ctx = FileContext::new();
        ctx.add(&sub).unwrap();
        let after_first = ctx.len();
        ctx.add(&sub).unwrap();

        assert_eq!(
            ctx.len(),
            after_first,
            "re-adding the same directory should add nothing new — \
             the count display relies on this to show \"(0 files)\""
        );
    }

    #[test]
    fn add_errors_on_missing_path() {
        let dir = tempdir();
        let missing = dir.join("nope.rs");

        let mut ctx = FileContext::new();
        let result = ctx.add(&missing);
        assert!(result.is_err(), "missing path should error, not silently no-op");
    }

    #[test]
    fn add_directory_recursively_includes_files() {
        let dir = tempdir();
        // Create two files directly inside the directory
        let a = dir.join("file_a.rs");
        let b = dir.join("file_b.rs");
        std::fs::write(&a, "// a\n").unwrap();
        std::fs::write(&b, "// b\n").unwrap();
        // Create a sub-directory with another file to verify recursion
        let sub = dir.join("nested");
        std::fs::create_dir_all(&sub).unwrap();
        let c = sub.join("file_c.rs");
        std::fs::write(&c, "// c\n").unwrap();

        let mut ctx = FileContext::new();
        ctx.add(&dir).unwrap();

        // All three files should be in the context
        for f in [&a, &b, &c] {
            assert!(
                ctx.paths.contains(&f.canonicalize().unwrap()),
                "expected {} to be in context",
                f.display()
            );
        }
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
