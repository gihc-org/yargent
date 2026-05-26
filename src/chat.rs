//! Interactive chat REPL.
//!
//! Drives a turn-by-turn conversation with the LLM: read a line from the user
//! via [`rustyline`], stream the model's reply to stdout, push both into the
//! history vector, scan for SEARCH/REPLACE edits, repeat. Slash-prefixed lines
//! (`/help`, `/add`, etc.) are intercepted before they reach the model.

use std::collections::{BTreeSet, HashSet};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use futures_util::StreamExt;
use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::FileHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Editor, Helper};
use tiktoken_rs::CoreBPE;

use crate::edit::{self, EditOutcome};
use crate::files::{self, FileContext};
use crate::git::Repo;
use crate::provider::{LLMProvider, Message};
use crate::repomap::RepoMap;

/// How many tokens of repo-map text we're willing to spend per request when
/// the user hasn't set their own budget. The rendered map is truncated to
/// fit; lowest-PageRanked files drop first.
const DEFAULT_MAP_TOKEN_BUDGET: usize = 1024;

/// Per-session settings derived from CLI flags + config file. Built in
/// `main.rs` and passed to [`run_chat`] verbatim.
pub struct SessionOptions {
    /// Skip auto-commit even when inside a git repo (CLI: `--no-commit`).
    pub no_commit: bool,
    /// Skip injecting the repo map at all (CLI: `--no-map`).
    pub no_map: bool,
    /// Token budget for the rendered repo map. `None` means use
    /// [`DEFAULT_MAP_TOKEN_BUDGET`].
    pub map_token_budget: Option<usize>,
    /// Whether the repo map is enabled by default for this session. The
    /// user can still toggle with `/map-on` / `/map-off` at runtime.
    pub map_default_enabled: bool,
    /// Skip auto-loading convention files this session (CLI: `--no-conventions`).
    pub no_conventions: bool,
    /// Whether convention auto-loading is enabled by default (from config).
    pub conventions_default_enabled: bool,
    /// Override list of convention filenames from config. `None` means use
    /// [`files::DEFAULT_CONVENTION_NAMES`].
    pub conventions_paths: Option<Vec<String>>,
}

/// Run the interactive chat loop until the user exits with `/quit` or Ctrl-D.
///
/// `system`, if provided, is concatenated *after* yargent's built-in coding
/// system prompt. Line-edit history persists to
/// `~/.local/share/yargent/history`.
pub async fn run_chat(
    provider: Box<dyn LLMProvider>,
    system: Option<String>,
    opts: SessionOptions,
) -> Result<()> {
    let tokenizer = tiktoken_rs::cl100k_base()?;
    let repo = if opts.no_commit {
        None
    } else {
        std::env::current_dir()
            .ok()
            .and_then(|cwd| Repo::discover(&cwd))
    };

    // Build the repo map up-front. For typical repo sizes this is a couple
    // hundred ms; for very large repos (10k+ files) it can be several seconds,
    // hence the progress line.
    let map_root = repo
        .as_ref()
        .map(|r| r.root().to_path_buf())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    print!(
        "yargent v{} ({}) — scanning {}... ",
        env!("CARGO_PKG_VERSION"),
        env!("YARGENT_GIT_SHA"),
        map_root.display()
    );
    io::stdout().flush().ok();
    let map_start = std::time::Instant::now();
    let repomap = RepoMap::build(&map_root);
    println!(
        "({} source file(s), {:.1}s)",
        repomap.len(),
        map_start.elapsed().as_secs_f32()
    );

    let map_enabled = opts.map_default_enabled && !opts.no_map;
    let map_token_budget = opts.map_token_budget.unwrap_or(DEFAULT_MAP_TOKEN_BUDGET);
    let mut state = ChatState::new(system, tokenizer, repo, repomap, map_enabled, map_token_budget);

    // Auto-load convention files from the repo root. Failures here are reported
    // inline but never abort startup — the chat REPL is still useful without
    // conventions loaded.
    if opts.conventions_default_enabled && !opts.no_conventions {
        let names_storage: Vec<String>;
        let names: Vec<&str> = match opts.conventions_paths {
            Some(custom) => {
                names_storage = custom;
                names_storage.iter().map(String::as_str).collect()
            }
            None => files::DEFAULT_CONVENTION_NAMES.to_vec(),
        };
        let found = files::discover_conventions(&map_root, &names);
        if !found.is_empty() {
            let mut added_names: Vec<String> = Vec::with_capacity(found.len());
            for path in &found {
                match state.files.add(path) {
                    Ok(canon) => added_names.push(short_name(&canon)),
                    Err(e) => eprintln!("conventions: skipping {}: {e:#}", path.display()),
                }
            }
            if !added_names.is_empty() {
                println!("conventions: loaded {}", added_names.join(", "));
            }
        }
    }

    let mut rl: Editor<MultiLineHelper, FileHistory> = Editor::new()?;
    rl.set_helper(Some(MultiLineHelper));
    let history_path = history_file_path();
    if let Some(ref p) = history_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = rl.load_history(p);
    }

    match &state.repo {
        Some(r) => println!("git repo: {}", r.root().display()),
        None => println!("no git repo (auto-commit disabled)"),
    }
    println!("type /help for commands, Ctrl-D to quit");

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
            Ok(full) => {
                // Parse and offer edits *before* pushing the assistant message
                // to history, so the diff output doesn't scroll away with the
                // model's prose.
                let applied = match handle_edits(&full) {
                    Ok(paths) => paths,
                    Err(e) => {
                        eprintln!("edit handling error: {e:#}");
                        Vec::new()
                    }
                };
                if !applied.is_empty()
                    && let Some(repo) = state.repo.as_ref()
                {
                    // Diagnose the diff before committing: lines/tests/pub items
                    // counts and a warning if AGENTS.md's "tests for new
                    // public API" rule looks violated. Best-effort — git
                    // failures inside diagnose come through as zeros, never
                    // block the commit.
                    let diag = crate::diagnostic::diagnose(repo, &applied);
                    print!("{}", diag.render());

                    // Only stop for confirmation when there's a concrete
                    // concern. Happy path stays one-line-summary-then-commit.
                    let proceed = if diag.has_concerns() {
                        let answer = tokio::task::block_in_place(|| {
                            prompt_line("commit anyway? [Y/n]: ")
                        })
                        .unwrap_or_default();
                        let t = answer.trim();
                        t.is_empty()
                            || t.eq_ignore_ascii_case("y")
                            || t.eq_ignore_ascii_case("yes")
                    } else {
                        true
                    };

                    if proceed {
                        let prompt = state
                            .history
                            .last()
                            .map(|m| m.content.as_str())
                            .unwrap_or("");
                        let msg = crate::git::build_commit_message(prompt, &applied);
                        match repo.commit_paths(&applied, &msg) {
                            Ok(sha) => println!("  ✓ committed {}", &sha[..7.min(sha.len())]),
                            Err(e) => eprintln!("  ✗ auto-commit failed: {e:#}"),
                        }
                    } else {
                        println!(
                            "  ✗ commit skipped — files remain edited in your working tree. \
                             Re-prompt the model to add tests, or commit manually with git."
                        );
                    }
                }
                state.history.push(Message::assistant(full));
            }
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
    /// User-supplied `--system` text, kept separate from `history` so that
    /// `/clear` doesn't touch it and so we can prepend yargent's built-in
    /// system prompt to it in `build_outgoing`.
    user_system: Option<String>,
    /// User/assistant turns only — no system messages, no synthetic injections.
    /// This is the clean record of what the user typed.
    history: Vec<Message>,
    /// Files the user has shared with `/add`. Re-read on every turn.
    files: FileContext,
    /// BPE tokenizer used for `/tokens` estimates. Loaded once; the merge
    /// tables are bundled into the binary so this never touches the network.
    tokenizer: CoreBPE,
    /// Discovered git working tree, if cwd is inside one. `None` disables
    /// auto-commit and `/undo`.
    repo: Option<Repo>,
    /// PageRank-ranked symbol overview of the repo. Built once at startup; can
    /// be rebuilt via `/map-rebuild` if files change significantly.
    repomap: RepoMap,
    /// Whether to inject the rendered repo map on each outgoing request.
    /// Toggled with `/map-on` / `/map-off`.
    map_enabled: bool,
    /// Per-session token budget for the rendered repo map. Resolved at
    /// startup from CLI/config; held as a field so the budget survives
    /// `/map-rebuild`.
    map_token_budget: usize,
}

impl ChatState {
    fn new(
        user_system: Option<String>,
        tokenizer: CoreBPE,
        repo: Option<Repo>,
        repomap: RepoMap,
        map_enabled: bool,
        map_token_budget: usize,
    ) -> Self {
        Self {
            user_system,
            history: Vec::new(),
            files: FileContext::new(),
            tokenizer,
            repo,
            repomap,
            map_enabled,
            map_token_budget,
        }
    }

    /// Build the actual list of messages to send to the provider this turn.
    ///
    /// Layout:
    /// 1. yargent's coding system prompt, with the user's `--system` appended
    ///    if one was supplied.
    /// 2. (optional) synthetic context message bundling repo map (top of
    ///    file) + `/add`ed file contents (below), plus a one-line assistant
    ///    acknowledgement. The bundle is only emitted when at least one
    ///    component is non-empty.
    /// 3. The actual user/assistant history.
    ///
    /// Step 2 is regenerated every turn so file edits propagate and the repo
    /// map's personalization tracks the current `/add` set. None of it ever
    /// appears in `self.history`.
    fn build_outgoing(&self) -> Result<Vec<Message>> {
        let mut out = Vec::with_capacity(self.history.len() + 3);

        let mut sys = String::from(edit::SYSTEM_PROMPT);
        if let Some(ref user_sys) = self.user_system {
            sys.push_str("\n\n");
            sys.push_str(user_sys);
        }
        out.push(Message::system(sys));

        // Build the synthetic context: repo map first (gives the model a
        // high-level overview), then the verbatim contents of /add'ed files.
        let focused_paths: HashSet<PathBuf> = self.files.paths().cloned().collect();
        let mut context = String::new();

        if self.map_enabled && !self.repomap.is_empty() {
            let rendered =
                self.repomap
                    .render(&focused_paths, self.map_token_budget, &self.tokenizer);
            if !rendered.is_empty() {
                context.push_str(&rendered);
                if !context.ends_with('\n') {
                    context.push('\n');
                }
                context.push('\n');
            }
        }

        if !self.files.is_empty() {
            context.push_str(&self.files.render()?);
        }

        if !context.is_empty() {
            out.push(Message::user(context));
            out.push(Message::assistant("Got it. I'll use this context."));
        }

        out.extend_from_slice(&self.history);
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
    let mut stdout = io::stdout();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        print!("{chunk}");
        stdout.flush().ok();
        full.push_str(&chunk);
    }
    println!();
    Ok(full)
}

/// Scan the model's reply for SEARCH/REPLACE blocks and walk the user through
/// approving each one. Returns the deduped, sorted set of paths that were
/// successfully written — the caller uses this list for auto-commit.
///
/// Individual edit failures (SEARCH not found, ambiguous match) are reported
/// inline and skipped. Only IO errors on the confirmation prompt itself bubble
/// up as `Err`.
fn handle_edits(reply: &str) -> Result<Vec<PathBuf>> {
    let edits = edit::parse_edits(reply);
    if edits.is_empty() {
        return Ok(Vec::new());
    }

    println!("\n--- {} edit(s) suggested ---", edits.len());

    let mut applied: BTreeSet<PathBuf> = BTreeSet::new();
    let mut auto_apply_rest = false;
    for (i, e) in edits.iter().enumerate() {
        println!("\n[{}/{}] {}", i + 1, edits.len(), e.path.display());
        print!("{}", edit::render_diff(e));

        let approve = if auto_apply_rest {
            true
        } else {
            let choice = tokio::task::block_in_place(|| {
                prompt_line("apply? [y]es / [n]o / [a]ll / [q]uit-prompt: ")
            })?;
            match choice.trim() {
                "y" | "Y" | "yes" => true,
                "a" | "A" | "all" => {
                    auto_apply_rest = true;
                    true
                }
                "q" | "Q" | "quit" => {
                    println!("(remaining {} edit(s) skipped)", edits.len() - i);
                    return Ok(applied.into_iter().collect());
                }
                _ => false,
            }
        };

        if !approve {
            println!("  skipped");
            continue;
        }

        match edit::apply_edit(e) {
            Ok(EditOutcome::Applied) => {
                println!("  ✓ applied");
                if let Ok(canon) = e.path.canonicalize() {
                    applied.insert(canon);
                }
            }
            Ok(EditOutcome::Created) => {
                println!("  ✓ created");
                if let Ok(canon) = e.path.canonicalize() {
                    applied.insert(canon);
                }
            }
            Ok(EditOutcome::NotFound) => {
                eprintln!("  ✗ SEARCH text not found in {}", e.path.display());
            }
            Ok(EditOutcome::Ambiguous(n)) => {
                eprintln!(
                    "  ✗ SEARCH text matches {n} places in {} — skipped",
                    e.path.display()
                );
            }
            Err(err) => eprintln!("  ✗ {err:#}"),
        }
    }
    Ok(applied.into_iter().collect())
}

fn prompt_line(msg: &str) -> Result<String> {
    let mut stdout = io::stdout();
    stdout.write_all(msg.as_bytes())?;
    stdout.flush()?;
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(line)
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
            state.history.clear();
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
                    // Snapshot the count before so we can report how many
                    // new files a directory walk actually produced.
                    // Re-adding the same dir yields "(0 files)" which is
                    // the honest answer and helps spot mistakes.
                    let before = state.files.len();
                    match state.files.add(path) {
                        Ok(canon) => {
                            let added = state.files.len() - before;
                            if canon.is_dir() {
                                println!("[added {} ({} files)]", canon.display(), added);
                            } else {
                                println!("[added {}]", canon.display());
                            }
                        }
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
        "map" => {
            if state.repomap.is_empty() {
                println!("[repo map is empty — no source files detected]");
                return false;
            }
            let focused: HashSet<PathBuf> = state.files.paths().cloned().collect();
            let rendered =
                state
                    .repomap
                    .render(&focused, state.map_token_budget, &state.tokenizer);
            print!("{rendered}");
        }
        "map-on" => {
            state.map_enabled = true;
            println!("[repo map injection enabled]");
        }
        "map-off" => {
            state.map_enabled = false;
            println!("[repo map injection disabled]");
        }
        "map-rebuild" => {
            let root = state
                .repo
                .as_ref()
                .map(|r| r.root().to_path_buf())
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_else(|| PathBuf::from("."));
            print!("[rebuilding map at {}... ", root.display());
            io::stdout().flush().ok();
            let start = std::time::Instant::now();
            state.repomap = crate::repomap::RepoMap::build(&root);
            println!(
                "{} files, {:.1}s]",
                state.repomap.len(),
                start.elapsed().as_secs_f32()
            );
        }
        "undo" => {
            let Some(repo) = state.repo.as_ref() else {
                println!("[no git repo — nothing to undo]");
                return false;
            };
            if !repo.head_is_yargent_commit() {
                println!("[HEAD is not a yargent commit — refusing to undo]");
                println!(
                    "(only yargent's auto-commits can be rolled back; any manual commits at HEAD must be undone with git directly)"
                );
                return false;
            }
            match repo.reset_to_parent() {
                Ok(sha) => println!("[undone — dropped {}]", &sha[..7.min(sha.len())]),
                Err(e) => println!("error: {e:#}"),
            }
        }
        "help" | "?" => print_help(),
        _ => println!("unknown command: /{cmd}  (try /help)"),
    }
    false
}

fn print_help() {
    println!("commands:");
    println!("  /add <path> [path...]   share files or directories with the model");
    println!("  /drop <path> [path...]  stop sharing files");
    println!("  /files                  list currently shared files");
    println!("  /tokens                 estimate tokens in next request");
    println!("  /map                    print the current repo map");
    println!("  /map-on /map-off        toggle injecting the map on each turn");
    println!("  /map-rebuild            re-walk and re-parse the repo");
    println!("  /undo                   roll back the most recent yargent auto-commit");
    println!("  /clear                  clear chat history (system + files preserved)");
    println!("  /history                print full conversation");
    println!("  /help                   show this help");
    println!("  /quit                   exit (or Ctrl-D)");
    println!();
    println!("editing:");
    println!("  When the model proposes edits as SEARCH/REPLACE blocks, yargent");
    println!("  shows a unified diff and prompts y/n/a/q before applying.");
    println!("  Successful edits are auto-committed in one git commit per turn");
    println!("  when run inside a git repo.");
    println!();
    println!("multi-line input:");
    println!("  Type `{{` on its own line to open a block; `}}` on its own line");
    println!("  closes and submits. Or end a line with `\\` to continue on the");
    println!("  next line. Single-line input still submits on Enter as before.");
}

fn history_file_path() -> Option<PathBuf> {
    dirs::data_local_dir().map(|p| p.join("yargent").join("history"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-line input helper
// ─────────────────────────────────────────────────────────────────────────────
//
// rustyline's `Validator` trait is the official entry point for multi-line
// input: each time the user presses Enter, rustyline asks the validator
// whether the buffer is "complete." If we return `Incomplete`, rustyline
// inserts a literal newline and lets the user keep typing; if `Valid`, the
// line is submitted as one input string (with embedded newlines).
//
// We trigger continuation on two patterns:
//
//   - **Block input** (aider-style): a line that is *exactly* `{` (whitespace-
//     trimmed) opens a multi-line block. A matching line of just `}` closes
//     it. The constraint that the brace must be alone on its line keeps the
//     parser from misfiring on pasted code that happens to contain braces.
//
//   - **Backslash continuation**: a line ending in `\` (literal backslash)
//     continues to the next line, like shell. Useful for one-off line
//     extensions without committing to a full block.
//
// Both markers stay in the submitted text — the model sees the braces and
// trailing backslashes verbatim. They're harmless and arguably useful as
// structure hints. If that becomes annoying we can post-process before
// pushing to history.

/// Empty struct implementing rustyline's `Helper` trait composite. The actual
/// behavior is in [`Validator`]; completer/highlighter/hinter take their
/// defaults (which do nothing).
#[derive(Default)]
struct MultiLineHelper;

impl Helper for MultiLineHelper {}
impl Completer for MultiLineHelper {
    type Candidate = String;
}
impl Hinter for MultiLineHelper {
    type Hint = String;
}
impl Highlighter for MultiLineHelper {}
impl Validator for MultiLineHelper {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        if needs_continuation(ctx.input()) {
            Ok(ValidationResult::Incomplete)
        } else {
            Ok(ValidationResult::Valid(None))
        }
    }
}

/// Decide whether `input` is still mid-multi-line and should accept another
/// Enter as a newline rather than a submission. See module-level comment for
/// the rules.
fn needs_continuation(input: &str) -> bool {
    let mut depth: i32 = 0;
    for line in input.lines() {
        match line.trim() {
            "{" => depth += 1,
            "}" => depth -= 1,
            _ => {}
        }
    }
    if depth > 0 {
        return true;
    }
    matches!(input.lines().last(), Some(l) if l.ends_with('\\'))
}

#[cfg(test)]
mod tests {
    use super::needs_continuation;

    #[test]
    fn single_line_submits() {
        assert!(!needs_continuation("hello"));
        assert!(!needs_continuation(""));
        assert!(!needs_continuation("/help"));
    }

    #[test]
    fn open_block_requires_more_input() {
        assert!(needs_continuation("{"));
        assert!(needs_continuation("{\nsome stuff"));
        assert!(needs_continuation("{\nline 1\nline 2"));
    }

    #[test]
    fn balanced_block_submits() {
        assert!(!needs_continuation("{\nstuff\n}"));
        assert!(!needs_continuation("{\nline 1\nline 2\n}"));
    }

    #[test]
    fn nested_blocks_track_depth() {
        assert!(needs_continuation("{\n{\nstuff\n}"));
        assert!(!needs_continuation("{\n{\nstuff\n}\n}"));
    }

    #[test]
    fn braces_with_other_content_do_not_count() {
        // A brace embedded in real content (pasted code) must NOT trigger
        // block mode — only lines that are *exactly* `{` or `}` count.
        assert!(!needs_continuation("fn foo() {"));
        assert!(!needs_continuation("fn foo() {\nbar()\n}"));
        assert!(!needs_continuation("{some text}"));
    }

    #[test]
    fn brace_with_surrounding_whitespace_still_counts() {
        // We trim before matching, so "  {  " is the same as "{".
        assert!(needs_continuation("   {\nstuff"));
        assert!(!needs_continuation("   {\nstuff\n   }   "));
    }

    #[test]
    fn trailing_backslash_continues() {
        assert!(needs_continuation("first line\\"));
        assert!(needs_continuation("first\\\nsecond\\"));
    }

    #[test]
    fn trailing_backslash_only_matters_on_last_line() {
        // A backslash inside the input but not at the very end shouldn't
        // hold up submission.
        assert!(!needs_continuation("middle\\\nfinal line"));
    }
}

/// Trim a path to just its filename for compact status lines. Falls back to
/// the full path if there's no filename component (rare — root directories
/// and similar edge cases).
fn short_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
