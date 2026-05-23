//! Parsing and applying SEARCH/REPLACE edit blocks.
//!
//! The wire format the model produces:
//!
//! ```text
//! path/to/file.rs
//! <<<<<<< SEARCH
//! old code
//! =======
//! new code
//! >>>>>>> REPLACE
//! ```
//!
//! - The filename is the most recent non-empty, non-fence line before
//!   `<<<<<<< SEARCH`. We accept paths wrapped in backticks (\``src/main.rs`\`)
//!   because models often do that.
//! - The SEARCH section must match the existing file content exactly, byte for
//!   byte; we use a literal substring search rather than fuzzy matching.
//! - An empty SEARCH section means "create this file with the REPLACE content
//!   as its initial body."
//! - Multiple blocks in one response are applied independently and in order.

use std::path::PathBuf;

use anyhow::{Context, Result};
use similar::{ChangeTag, TextDiff};

/// System prompt that teaches the model the SEARCH/REPLACE format.
///
/// Injected as the first system message on every chat completion (before any
/// user-supplied `--system` text). Without it the model has no way to know
/// what edit format we expect.
pub const SYSTEM_PROMPT: &str = "\
You are yargent, an AI coding assistant working in the user's terminal.

When the user asks you to modify code, output the changes as SEARCH/REPLACE \
blocks in this EXACT format:

path/to/file.rs
<<<<<<< SEARCH
existing code, copied verbatim
=======
new code that replaces it
>>>>>>> REPLACE

Format rules — break any of these and your edit will be rejected:
- The path goes on its own line immediately before <<<<<<< SEARCH, with no \
backticks, no list markers, no leading whitespace.
- The SEARCH section must match the current file content byte for byte, \
including indentation and blank lines. Include just enough surrounding context \
to make the match unique within the file — too little context risks ambiguity, \
too much wastes tokens.
- To create a new file, leave the SEARCH section empty.
- One block edits one location. Emit multiple blocks to make multiple edits; \
they apply independently and in order.
- Briefly explain in prose what you are changing and why, then provide the \
blocks. Don't bury the blocks inside extra Markdown fences.

If you only need to discuss code without modifying it, skip the blocks \
entirely and reply with prose. The user will ask explicitly when they want a \
change applied.
";

/// A single SEARCH/REPLACE edit extracted from a model response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit {
    pub path: PathBuf,
    pub search: String,
    pub replace: String,
}

impl Edit {
    /// True if this edit creates a new file rather than modifying an existing one.
    pub fn is_creation(&self) -> bool {
        self.search.trim().is_empty()
    }
}

/// Extract every well-formed SEARCH/REPLACE block from `text`.
///
/// Malformed blocks (no preceding filename, unterminated section) are silently
/// skipped — they can't be applied safely and surfacing them as errors would
/// make the chat loop noisy when the model is just discussing code.
pub fn parse_edits(text: &str) -> Vec<Edit> {
    let lines: Vec<&str> = text.lines().collect();
    let mut edits = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].trim() != "<<<<<<< SEARCH" {
            i += 1;
            continue;
        }

        let path = find_filename(&lines[..i]);
        i += 1;

        let search_start = i;
        while i < lines.len() && lines[i].trim() != "=======" {
            i += 1;
        }
        if i >= lines.len() {
            break; // unterminated: SEARCH never closed
        }
        let search = join_lines(&lines[search_start..i]);
        i += 1; // consume "======="

        let replace_start = i;
        while i < lines.len() && lines[i].trim() != ">>>>>>> REPLACE" {
            i += 1;
        }
        if i >= lines.len() {
            break; // unterminated: REPLACE never closed
        }
        let replace = join_lines(&lines[replace_start..i]);
        i += 1; // consume ">>>>>>> REPLACE"

        if let Some(p) = path {
            edits.push(Edit {
                path: p,
                search,
                replace,
            });
        }
    }

    edits
}

/// Walk backwards from a SEARCH marker to find the filename. Skip empty lines
/// and Markdown fences; strip optional surrounding backticks.
fn find_filename(prev: &[&str]) -> Option<PathBuf> {
    for raw in prev.iter().rev() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("```") {
            continue;
        }
        let cleaned = line.trim_matches('`').trim();
        if cleaned.is_empty() {
            return None;
        }
        return Some(PathBuf::from(cleaned));
    }
    None
}

/// Join SEARCH/REPLACE body lines back into a single string.
///
/// `&str::lines()` strips line terminators, so we re-introduce a single `\n`
/// between rows. The result deliberately has no trailing `\n` — that matches
/// the typical case where the SEARCH text is followed by more file content,
/// and `replace` keeps the same shape so substitution doesn't shift newlines.
fn join_lines(lines: &[&str]) -> String {
    lines.join("\n")
}

/// Outcome of applying a single edit. The chat loop uses this to summarize
/// what happened after a confirmation prompt.
#[derive(Debug)]
pub enum EditOutcome {
    Applied,
    Created,
    NotFound,
    Ambiguous(usize),
}

/// Apply `edit` to disk.
///
/// - Empty SEARCH creates a new file (with parent directories), erroring if
///   the file already exists — we don't want to silently overwrite.
/// - Otherwise the SEARCH text must occur exactly once in the file; we then
///   replace that single occurrence.
pub fn apply_edit(edit: &Edit) -> Result<EditOutcome> {
    if edit.is_creation() {
        if edit.path.exists() {
            anyhow::bail!(
                "refusing to create {}: file already exists (use a SEARCH section to modify it)",
                edit.path.display()
            );
        }
        if let Some(parent) = edit.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating parent of {}", edit.path.display()))?;
        }
        std::fs::write(&edit.path, &edit.replace)
            .with_context(|| format!("writing {}", edit.path.display()))?;
        return Ok(EditOutcome::Created);
    }

    let current = std::fs::read_to_string(&edit.path)
        .with_context(|| format!("reading {}", edit.path.display()))?;

    let count = current.matches(&edit.search).count();
    match count {
        0 => Ok(EditOutcome::NotFound),
        1 => {
            let new_content = current.replacen(&edit.search, &edit.replace, 1);
            std::fs::write(&edit.path, new_content)
                .with_context(|| format!("writing {}", edit.path.display()))?;
            Ok(EditOutcome::Applied)
        }
        n => Ok(EditOutcome::Ambiguous(n)),
    }
}

const ANSI_RED: &str = "\x1b[31m";
const ANSI_GREEN: &str = "\x1b[32m";
const ANSI_CYAN: &str = "\x1b[36m";
const ANSI_RESET: &str = "\x1b[0m";

/// Render an `Edit` as a colorized unified diff for the terminal.
///
/// For modifications we diff the current file against what it would look like
/// after applying. For creations we just show every line as an addition with
/// the empty file as the baseline.
pub fn render_diff(edit: &Edit) -> String {
    let header = format!(
        "{ANSI_CYAN}--- {0}\n+++ {0}{ANSI_RESET}\n",
        edit.path.display()
    );

    let (old, new) = if edit.is_creation() {
        (String::new(), edit.replace.clone())
    } else {
        let current = std::fs::read_to_string(&edit.path).unwrap_or_default();
        let count = current.matches(&edit.search).count();
        if count != 1 {
            // We still preview by diffing the bare SEARCH against REPLACE so the
            // user sees the intent. apply_edit() will surface the real error.
            (edit.search.clone(), edit.replace.clone())
        } else {
            let proposed = current.replacen(&edit.search, &edit.replace, 1);
            (current, proposed)
        }
    };

    let diff = TextDiff::from_lines(&old, &new);
    let mut out = header;
    for change in diff.iter_all_changes() {
        let (sign, color) = match change.tag() {
            ChangeTag::Delete => ("-", ANSI_RED),
            ChangeTag::Insert => ("+", ANSI_GREEN),
            ChangeTag::Equal => (" ", ""),
        };
        out.push_str(color);
        out.push_str(sign);
        out.push_str(change.value());
        if !change.value().ends_with('\n') {
            out.push('\n');
        }
        if !color.is_empty() {
            out.push_str(ANSI_RESET);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn parse_single_block() {
        let text = "\
Here's the fix:

src/main.rs
<<<<<<< SEARCH
fn old() {}
=======
fn new() {}
>>>>>>> REPLACE
";
        let edits = parse_edits(text);
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].path, PathBuf::from("src/main.rs"));
        assert_eq!(edits[0].search, "fn old() {}");
        assert_eq!(edits[0].replace, "fn new() {}");
    }

    #[test]
    fn parse_multiple_blocks() {
        let text = "\
a.rs
<<<<<<< SEARCH
1
=======
2
>>>>>>> REPLACE

b.rs
<<<<<<< SEARCH
3
=======
4
>>>>>>> REPLACE
";
        let edits = parse_edits(text);
        assert_eq!(edits.len(), 2);
        assert_eq!(edits[0].path, PathBuf::from("a.rs"));
        assert_eq!(edits[1].path, PathBuf::from("b.rs"));
    }

    #[test]
    fn parse_strips_backticks_around_filename() {
        let text = "\
`src/foo.rs`
<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE
";
        let edits = parse_edits(text);
        assert_eq!(edits[0].path, PathBuf::from("src/foo.rs"));
    }

    #[test]
    fn parse_ignores_unterminated_block() {
        let text = "\
src/main.rs
<<<<<<< SEARCH
incomplete
";
        assert!(parse_edits(text).is_empty());
    }

    #[test]
    fn apply_modifies_existing_file() {
        let dir = tempdir();
        let path = dir.join("a.txt");
        std::fs::write(&path, "hello world\n").unwrap();

        let edit = Edit {
            path: path.clone(),
            search: "hello".to_string(),
            replace: "goodbye".to_string(),
        };
        let outcome = apply_edit(&edit).unwrap();
        assert!(matches!(outcome, EditOutcome::Applied));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "goodbye world\n");
    }

    #[test]
    fn apply_creates_new_file_with_empty_search() {
        let dir = tempdir();
        let path = dir.join("sub/new.txt");

        let edit = Edit {
            path: path.clone(),
            search: String::new(),
            replace: "fresh content\n".to_string(),
        };
        let outcome = apply_edit(&edit).unwrap();
        assert!(matches!(outcome, EditOutcome::Created));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "fresh content\n");
    }

    #[test]
    fn apply_reports_ambiguous_when_search_matches_multiple_times() {
        let dir = tempdir();
        let path = dir.join("a.txt");
        std::fs::write(&path, "x\nx\n").unwrap();

        let edit = Edit {
            path,
            search: "x".to_string(),
            replace: "y".to_string(),
        };
        let outcome = apply_edit(&edit).unwrap();
        assert!(matches!(outcome, EditOutcome::Ambiguous(2)));
    }

    #[test]
    fn apply_reports_not_found_when_search_missing() {
        let dir = tempdir();
        let path = dir.join("a.txt");
        std::fs::write(&path, "hello\n").unwrap();

        let edit = Edit {
            path,
            search: "missing".to_string(),
            replace: "anything".to_string(),
        };
        let outcome = apply_edit(&edit).unwrap();
        assert!(matches!(outcome, EditOutcome::NotFound));
    }

    fn tempdir() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("yargent-edit-test-{pid}-{n}"));
        std::fs::create_dir_all(&dir).unwrap();
        // Suppress unused-Write warning from doctest-style construction
        let _ = std::fs::File::create(dir.join(".keep")).map(|mut f| f.write_all(b""));
        dir
    }
}
