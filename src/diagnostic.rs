//! Post-edit diagnostic: count lines, `#[test]` attributes, and new `pub`
//! items in the working-tree diff so we can warn the user when AGENTS.md's
//! "test every edge case" rule was likely violated.
//!
//! Runs after edits have been applied to disk but **before** the auto-commit.
//! The user always sees a one-line summary per file; we only prompt for
//! confirmation when a concerning pattern is detected (new public API with
//! no new tests). Otherwise the commit proceeds silently as before.
//!
//! All git interaction is best-effort — if a `git diff` invocation fails for
//! any reason, that file's counts just come through as zero and the commit
//! still goes ahead. We never block the commit on a diagnostic failure
//! itself.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::git::Repo;

/// Aggregated diagnostic across a batch of just-applied edits.
#[derive(Debug, Default)]
pub struct Diagnostic {
    pub files: Vec<FileStats>,
    pub total_added: u64,
    pub total_removed: u64,
    pub new_test_attrs: u64,
    pub new_pub_items: u64,
}

/// Per-file slice of [`Diagnostic`].
#[derive(Debug)]
pub struct FileStats {
    pub path: PathBuf,
    pub added: u64,
    pub removed: u64,
    pub tests_added: u64,
}

impl Diagnostic {
    /// True if the diff adds new public API without any new tests — the
    /// canonical AGENTS.md "missing edge-case tests" violation.
    pub fn has_concerns(&self) -> bool {
        self.new_pub_items > 0 && self.new_test_attrs == 0
    }

    /// Compact summary printed before commit: one line per file plus any
    /// warnings. Always non-empty when there are files, even on the happy
    /// path — the visibility is the feature.
    pub fn render(&self) -> String {
        if self.files.is_empty() {
            return String::new();
        }
        let mut out = String::from("\nedit summary:\n");
        for f in &self.files {
            let tests_suffix = if f.tests_added > 0 {
                format!(", +{} test(s)", f.tests_added)
            } else {
                String::new()
            };
            out.push_str(&format!(
                "  {}: +{} -{}{}\n",
                f.path.display(),
                f.added,
                f.removed,
                tests_suffix
            ));
        }
        if self.has_concerns() {
            out.push_str(&format!(
                "  ⚠ {} new public item(s) added without tests\n",
                self.new_pub_items
            ));
            out.push_str(
                "    AGENTS.md says: cover edge cases with tests in the same commit.\n",
            );
        }
        out
    }
}

/// Run the diagnostic for `paths` against the working tree of `repo`.
pub fn diagnose(repo: &Repo, paths: &[PathBuf]) -> Diagnostic {
    let mut diag = Diagnostic::default();
    for p in paths {
        let (added, removed) = git_numstat(repo, p).unwrap_or((0, 0));
        let diff_text = git_diff(repo, p).unwrap_or_default();
        let tests_added = count_added_lines_matching(&diff_text, is_test_attr_line);
        let pubs_added = count_added_lines_matching(&diff_text, is_pub_item_line);

        diag.total_added += added;
        diag.total_removed += removed;
        diag.new_test_attrs += tests_added;
        diag.new_pub_items += pubs_added;
        diag.files.push(FileStats {
            path: p.clone(),
            added,
            removed,
            tests_added,
        });
    }
    diag
}

// ─────────────────────────────────────────────────────────────────────────────
// Git subprocess helpers
// ─────────────────────────────────────────────────────────────────────────────

fn git_numstat(repo: &Repo, path: &Path) -> Option<(u64, u64)> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo.root())
        .args(["diff", "HEAD", "--numstat", "--"])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = std::str::from_utf8(&out.stdout).ok()?;
    parse_numstat(s)
}

fn git_diff(repo: &Repo, path: &Path) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo.root())
        .args(["diff", "HEAD", "--"])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure parsing — testable in isolation without touching git
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `git diff --numstat` output. Format: `<added>\t<removed>\t<path>`.
/// Returns `None` for binary files (which use `-\t-\t<path>`) or anything
/// we can't parse cleanly — caller treats that as "0 lines added".
fn parse_numstat(s: &str) -> Option<(u64, u64)> {
    let line = s.lines().next()?;
    let mut parts = line.split('\t');
    let added: u64 = parts.next()?.parse().ok()?;
    let removed: u64 = parts.next()?.parse().ok()?;
    Some((added, removed))
}

/// Count diff lines that start with `+` (added) — excluding the `+++ b/...`
/// header — whose content matches `pred`.
fn count_added_lines_matching<F: Fn(&str) -> bool>(diff: &str, pred: F) -> u64 {
    diff.lines()
        .filter(|l| l.starts_with('+') && !l.starts_with("+++"))
        // Strip the leading '+' before passing to the predicate so callers
        // don't have to know about the diff format.
        .filter(|l| pred(&l[1..]))
        .count() as u64
}

fn is_test_attr_line(line: &str) -> bool {
    line.trim_start().starts_with("#[test]")
}

fn is_pub_item_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("pub fn ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("pub trait ")
        || trimmed.starts_with("pub mod ")
        || trimmed.starts_with("pub const ")
        || trimmed.starts_with("pub static ")
        || trimmed.starts_with("pub type ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_numstat_extracts_counts() {
        assert_eq!(parse_numstat("12\t3\tsrc/foo.rs\n"), Some((12, 3)));
        assert_eq!(parse_numstat("0\t0\tsrc/foo.rs\n"), Some((0, 0)));
    }

    #[test]
    fn parse_numstat_rejects_garbage() {
        assert_eq!(parse_numstat(""), None);
        assert_eq!(parse_numstat("not a numstat"), None);
        // Binary files appear as "-\t-\tpath" — should be None, not a parse.
        assert_eq!(parse_numstat("-\t-\timage.png\n"), None);
    }

    #[test]
    fn count_added_lines_finds_test_attrs() {
        let diff = "\
diff --git a/x.rs b/x.rs
+++ b/x.rs
@@ -1 +1,3 @@
 fn foo() {}
+#[test]
+fn test_foo() {}
";
        assert_eq!(count_added_lines_matching(diff, is_test_attr_line), 1);
    }

    #[test]
    fn count_added_lines_ignores_context_and_removed() {
        // Lines starting with space (context) or '-' (removed) must not count
        // even when their content matches the predicate.
        let diff = "\
@@ -1,3 +1 @@
-#[test]
-fn old_test() {}
 #[test]
";
        assert_eq!(count_added_lines_matching(diff, is_test_attr_line), 0);
    }

    #[test]
    fn count_added_lines_ignores_diff_file_header() {
        // The "+++ b/path" header starts with '+' but is structural — it
        // shouldn't match arbitrary predicates.
        let diff = "\
+++ b/x.rs
@@ -0,0 +1,2 @@
+pub fn foo() {}
+pub fn bar() {}
";
        assert_eq!(count_added_lines_matching(diff, is_pub_item_line), 2);
    }

    #[test]
    fn pub_item_predicate_handles_indentation() {
        // Pub items inside a module are indented; trim before matching.
        assert!(is_pub_item_line("pub fn top_level() {}"));
        assert!(is_pub_item_line("    pub fn indented() {}"));
        assert!(is_pub_item_line("\tpub struct Foo;"));
        // Non-pub items shouldn't match.
        assert!(!is_pub_item_line("fn private() {}"));
        assert!(!is_pub_item_line("let pub_count = 0;"));
    }

    #[test]
    fn has_concerns_only_when_pub_added_without_tests() {
        let baseline = Diagnostic::default();
        assert!(!baseline.has_concerns(), "empty diff is fine");

        let pubs_with_tests = Diagnostic {
            new_pub_items: 2,
            new_test_attrs: 1,
            ..Default::default()
        };
        assert!(!pubs_with_tests.has_concerns());

        let pubs_no_tests = Diagnostic {
            new_pub_items: 2,
            new_test_attrs: 0,
            ..Default::default()
        };
        assert!(pubs_no_tests.has_concerns());

        let tests_no_pubs = Diagnostic {
            new_pub_items: 0,
            new_test_attrs: 3,
            ..Default::default()
        };
        assert!(!tests_no_pubs.has_concerns(), "tests-only diffs are fine");
    }

    #[test]
    fn render_includes_warning_when_concerns_present() {
        let diag = Diagnostic {
            files: vec![FileStats {
                path: PathBuf::from("x.rs"),
                added: 30,
                removed: 2,
                tests_added: 0,
            }],
            total_added: 30,
            total_removed: 2,
            new_test_attrs: 0,
            new_pub_items: 2,
        };
        let r = diag.render();
        assert!(r.contains("x.rs: +30 -2"));
        assert!(r.contains("⚠"));
        assert!(r.contains("AGENTS.md"));
    }

    #[test]
    fn render_is_empty_when_no_files() {
        assert!(Diagnostic::default().render().is_empty());
    }
}
