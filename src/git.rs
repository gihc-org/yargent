//! Git integration via subprocess.
//!
//! We shell out to the system `git` binary rather than using a Rust git
//! library. The trade-off behind that choice:
//!
//! - `git2` would link libgit2 (C) — exactly the kind of native dep we want to
//!   avoid on Termux/Android.
//! - `gix` is pure Rust but its write-side API (commit, reset) is still
//!   maturing and would be substantially more code than its read side.
//! - The `git` binary is a prerequisite for almost any project a user would
//!   point yargent at, and is one line to install in Termux (`pkg install git`).
//!
//! Subprocess overhead is invisible at the human latency we operate at here.
//! Worth revisiting only if we ever need to embed yargent somewhere without
//! a git binary on PATH.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use anyhow::{Context, Result};

/// A trailer line appended to every commit yargent creates. `/undo` refuses to
/// reset HEAD unless this trailer is present, so a manual `git commit` the user
/// made directly is never rolled back by accident.
pub const COMMIT_TRAILER: &str = "Yargent-Edit: yes";

/// Handle on a git working tree. Holds the absolute path to the repo root so
/// every subsequent `git` invocation can run with `-C <root>` regardless of
/// where yargent's own cwd ends up.
pub struct Repo {
    root: PathBuf,
}

impl Repo {
    /// Try to find a git repo by walking up from `start`. Returns `None` if:
    /// - `start` (or any ancestor) is not inside a git working tree, OR
    /// - the `git` binary isn't on PATH at all.
    ///
    /// Both failure modes look identical to the caller because the desired
    /// behavior is the same — disable git-related features and continue.
    pub fn discover(start: &Path) -> Option<Self> {
        let start = start.to_str()?;
        let output = Command::new("git")
            .args(["-C", start, "rev-parse", "--show-toplevel"])
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let root = String::from_utf8(output.stdout).ok()?.trim().to_string();
        if root.is_empty() {
            return None;
        }
        Some(Self {
            root: PathBuf::from(root),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn root_str(&self) -> &str {
        // Set in discover() from valid UTF-8; safe to unwrap.
        self.root.to_str().expect("repo root is valid UTF-8")
    }

    /// Stage the listed paths and create one commit with `message`.
    ///
    /// We pass each path explicitly to `git add` rather than `git add -A` so
    /// any unrelated dirty changes elsewhere in the repo stay out of the
    /// commit. Returns the full SHA of the new commit.
    pub fn commit_paths(&self, paths: &[PathBuf], message: &str) -> Result<String> {
        if paths.is_empty() {
            anyhow::bail!("nothing to commit");
        }

        let root = self.root_str();

        // Stage each path. Use one `git add -- <paths...>` invocation so it's
        // a single fork; the `--` guard prevents paths starting with `-` from
        // being interpreted as flags.
        let mut add = Command::new("git");
        add.args(["-C", root, "add", "--"]);
        for p in paths {
            add.arg(p);
        }
        let add_out = add
            .output()
            .context("failed to invoke `git add` (is git installed?)")?;
        if !add_out.status.success() {
            anyhow::bail!("git add failed: {}", stderr_or_status(&add_out));
        }

        // Did staging actually produce a diff? If the model emitted edits that
        // were no-ops (e.g. SEARCH and REPLACE identical) there is nothing to
        // commit and `git commit` would error out — short-circuit with a clear
        // message instead.
        let diff_status = Command::new("git")
            .args(["-C", root, "diff", "--cached", "--quiet"])
            .status()
            .context("failed to invoke `git diff`")?;
        if diff_status.success() {
            anyhow::bail!("no staged changes to commit (edits were no-ops)");
        }

        let commit_out = Command::new("git")
            .args(["-C", root, "commit", "-m", message])
            .output()
            .context("failed to invoke `git commit`")?;
        if !commit_out.status.success() {
            anyhow::bail!("git commit failed: {}", stderr_or_status(&commit_out));
        }

        self.head_sha()
    }

    /// Full SHA of HEAD.
    pub fn head_sha(&self) -> Result<String> {
        let out = Command::new("git")
            .args(["-C", self.root_str(), "rev-parse", "HEAD"])
            .output()
            .context("failed to invoke `git rev-parse`")?;
        if !out.status.success() {
            anyhow::bail!("git rev-parse HEAD failed: {}", stderr_or_status(&out));
        }
        Ok(String::from_utf8(out.stdout)
            .context("git rev-parse returned non-UTF8")?
            .trim()
            .to_string())
    }

    /// Full commit message body of HEAD. Used to decide whether `/undo` is safe.
    pub fn head_message(&self) -> Result<String> {
        let out = Command::new("git")
            .args(["-C", self.root_str(), "log", "-1", "--format=%B"])
            .output()
            .context("failed to invoke `git log`")?;
        if !out.status.success() {
            anyhow::bail!("git log failed: {}", stderr_or_status(&out));
        }
        Ok(String::from_utf8(out.stdout)
            .context("git log returned non-UTF8")?
            .trim_end()
            .to_string())
    }

    /// True if HEAD's commit message contains [`COMMIT_TRAILER`].
    pub fn head_is_yargent_commit(&self) -> bool {
        match self.head_message() {
            Ok(msg) => msg.lines().any(|l| l.trim() == COMMIT_TRAILER),
            Err(_) => false,
        }
    }

    /// Hard-reset HEAD to its parent, throwing away the most recent commit and
    /// its working-tree effects. Caller is responsible for confirming this is
    /// safe — typically by checking [`Self::head_is_yargent_commit`] first.
    /// Returns the SHA that was just dropped.
    pub fn reset_to_parent(&self) -> Result<String> {
        let dropped = self.head_sha()?;
        let status = Command::new("git")
            .args(["-C", self.root_str(), "reset", "--hard", "HEAD~1"])
            .status()
            .context("failed to invoke `git reset`")?;
        if !status.success() {
            anyhow::bail!("git reset --hard HEAD~1 failed");
        }
        Ok(dropped)
    }
}

/// Build a commit message from the user's prompt + the touched paths.
///
/// First line is the prompt's first line trimmed to 60 characters (with an
/// ellipsis if it was longer) — that gives reasonable subjects in `git log`.
/// The body lists every file touched and ends with [`COMMIT_TRAILER`] so
/// `/undo` can recognize the commit as ours.
pub fn build_commit_message(prompt: &str, paths: &[PathBuf]) -> String {
    let subject = compose_subject(prompt);
    let mut out = format!("yargent: {subject}\n\nFiles:\n");
    for p in paths {
        out.push_str("- ");
        out.push_str(&p.display().to_string());
        out.push('\n');
    }
    out.push('\n');
    out.push_str(COMMIT_TRAILER);
    out.push('\n');
    out
}

fn compose_subject(prompt: &str) -> String {
    let cleaned = first_meaningful_line(prompt).unwrap_or("apply edits");
    const LIMIT: usize = 60;
    if cleaned.chars().count() <= LIMIT {
        cleaned.to_string()
    } else {
        let truncated: String = cleaned.chars().take(LIMIT - 1).collect();
        format!("{truncated}…")
    }
}

/// Walk `prompt` looking for the first line that's actually content — i.e.
/// not blank, not a bare multi-line block marker (`{` / `}`), and not a bare
/// line-continuation backslash. Strips a trailing `\` from the chosen line
/// so the commit subject reads cleanly.
///
/// Returns `None` when the prompt is empty or contains only noise lines.
/// The caller substitutes a default ("apply edits") in that case.
fn first_meaningful_line(prompt: &str) -> Option<&str> {
    for line in prompt.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "{" || trimmed == "}" || trimmed == "\\" {
            continue;
        }
        // Strip trailing backslash continuation if present, then re-trim any
        // whitespace it was hiding.
        let cleaned = trimmed.strip_suffix('\\').unwrap_or(trimmed).trim_end();
        if cleaned.is_empty() {
            continue;
        }
        return Some(cleaned);
    }
    None
}

fn stderr_or_status(out: &Output) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr);
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        format!("exit {}", out.status)
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subject_trims_long_first_line() {
        let prompt = "a".repeat(120);
        let msg = build_commit_message(&prompt, &[PathBuf::from("x.rs")]);
        let subject = msg.lines().next().unwrap();
        // "yargent: " + 60 chars
        assert_eq!(subject.chars().count(), "yargent: ".len() + 60);
        assert!(subject.ends_with('…'));
    }

    #[test]
    fn subject_uses_only_first_line() {
        let msg = build_commit_message("first line\nsecond line", &[PathBuf::from("x")]);
        assert_eq!(msg.lines().next().unwrap(), "yargent: first line");
    }

    #[test]
    fn subject_falls_back_when_prompt_blank() {
        let msg = build_commit_message("   \n\n", &[PathBuf::from("x")]);
        assert_eq!(msg.lines().next().unwrap(), "yargent: apply edits");
    }

    #[test]
    fn subject_skips_opening_brace_from_multiline_block() {
        // The case that motivated this fix: multi-line block input means the
        // first line is literally "{", and we used to produce "yargent: {".
        let prompt = "{\nAdd the yargent version to the startup banner\nand also do X\n}";
        let msg = build_commit_message(prompt, &[PathBuf::from("x")]);
        assert_eq!(
            msg.lines().next().unwrap(),
            "yargent: Add the yargent version to the startup banner"
        );
    }

    #[test]
    fn subject_skips_lone_closing_brace_and_backslash() {
        // Closing `}` and bare backslash continuations are also noise.
        let prompt = "}\n\\\nReal content here";
        let msg = build_commit_message(prompt, &[PathBuf::from("x")]);
        assert_eq!(msg.lines().next().unwrap(), "yargent: Real content here");
    }

    #[test]
    fn subject_strips_trailing_backslash_continuation() {
        // Backslash continuation on the chosen line shouldn't end up in the
        // committed subject.
        let prompt = "first line of prompt \\\nrest of prompt";
        let msg = build_commit_message(prompt, &[PathBuf::from("x")]);
        assert_eq!(msg.lines().next().unwrap(), "yargent: first line of prompt");
    }

    #[test]
    fn subject_falls_back_when_only_block_markers() {
        // Pathological case: prompt is literally just block markers with no
        // content. The old code would have produced "yargent: {".
        let prompt = "{\n\n}";
        let msg = build_commit_message(prompt, &[PathBuf::from("x")]);
        assert_eq!(msg.lines().next().unwrap(), "yargent: apply edits");
    }

    #[test]
    fn message_includes_trailer_and_files() {
        let msg = build_commit_message(
            "fix bug",
            &[PathBuf::from("a.rs"), PathBuf::from("b.rs")],
        );
        assert!(msg.contains("- a.rs"));
        assert!(msg.contains("- b.rs"));
        assert!(msg.contains(COMMIT_TRAILER));
    }
}
