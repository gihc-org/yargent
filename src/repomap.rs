//! Repo map: a compact, PageRank-ranked symbol overview of the entire
//! repository, injected into the model's context so it knows what exists in
//! files the user hasn't explicitly `/add`ed.
//!
//! Pipeline:
//!   1. **Walk** the repo with `.gitignore` respect ([`ignore`] crate).
//!   2. **Parse** each source file with the appropriate tree-sitter grammar.
//!   3. Run a **definitions** query to extract symbols this file defines.
//!   4. Run a **references** query to extract symbols this file uses but
//!      doesn't define (function calls, type uses, etc.).
//!   5. **Build a directed graph**: an edge from file A to file B exists when
//!      A references a symbol B defines. The edge weight is the number of
//!      such references.
//!   6. **PageRank** the graph. A personalization vector biases the walk
//!      toward files the user has `/add`ed, so the map highlights what's
//!      most relevant to the current focus.
//!   7. **Render** the top-ranked files (excluding the ones the user already
//!      has in full), each with their top symbols, capped by a token budget.
//!
//! The whole pipeline is deterministic and re-run each turn — cheap enough at
//! the repo sizes a single developer typically works with that we don't need
//! incremental caching yet.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tiktoken_rs::CoreBPE;
use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator};

// ─────────────────────────────────────────────────────────────────────────────
// Language detection and tree-sitter queries
// ─────────────────────────────────────────────────────────────────────────────

/// Languages we know how to parse. Adding a new one means:
/// - extending [`Language::detect`] with the file extension,
/// - hooking the grammar crate up in [`Language::tree_sitter_language`],
/// - writing the two queries below in [`Language::definitions_query`]
///   and [`Language::references_query`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
}

impl Language {
    pub fn detect(path: &Path) -> Option<Self> {
        match path.extension().and_then(|s| s.to_str())? {
            "rs" => Some(Self::Rust),
            "py" => Some(Self::Python),
            "js" | "mjs" | "cjs" | "jsx" => Some(Self::JavaScript),
            "ts" | "tsx" => Some(Self::TypeScript),
            _ => None,
        }
    }

    fn tree_sitter_language(self) -> tree_sitter::Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        }
    }

    /// Tree-sitter query in S-expression form that captures top-level symbol
    /// definitions. The `@name` capture is the identifier we extract — every
    /// pattern must include exactly one.
    fn definitions_query(self) -> &'static str {
        match self {
            Self::Rust => "\
(function_item name: (identifier) @name)
(struct_item name: (type_identifier) @name)
(enum_item name: (type_identifier) @name)
(trait_item name: (type_identifier) @name)
(impl_item type: (type_identifier) @name)
(mod_item name: (identifier) @name)
(const_item name: (identifier) @name)
(static_item name: (identifier) @name)
(type_item name: (type_identifier) @name)
(macro_definition name: (identifier) @name)
",
            Self::Python => "\
(function_definition name: (identifier) @name)
(class_definition name: (identifier) @name)
",
            Self::JavaScript => "\
(function_declaration name: (identifier) @name)
(class_declaration name: (identifier) @name)
(method_definition name: (property_identifier) @name)
(variable_declarator name: (identifier) @name value: (arrow_function))
(variable_declarator name: (identifier) @name value: (function))
",
            Self::TypeScript => "\
(function_declaration name: (identifier) @name)
(class_declaration name: (type_identifier) @name)
(interface_declaration name: (type_identifier) @name)
(type_alias_declaration name: (type_identifier) @name)
(enum_declaration name: (identifier) @name)
(method_definition name: (property_identifier) @name)
",
        }
    }

    /// Query capturing identifier *uses* — things this file references but
    /// (likely) defines elsewhere. We intentionally cast a wide net here;
    /// false positives just don't draw a graph edge (the named symbol isn't
    /// defined anywhere else) and false negatives miss edges. Wide-and-noisy
    /// is the cheaper failure mode.
    fn references_query(self) -> &'static str {
        match self {
            Self::Rust => "\
(call_expression function: (identifier) @ref)
(call_expression function: (scoped_identifier name: (identifier) @ref))
(call_expression function: (field_expression field: (field_identifier) @ref))
(type_identifier) @ref
",
            Self::Python => "\
(call function: (identifier) @ref)
(call function: (attribute attribute: (identifier) @ref))
",
            Self::JavaScript => "\
(call_expression function: (identifier) @ref)
(call_expression function: (member_expression property: (property_identifier) @ref))
(new_expression constructor: (identifier) @ref)
",
            Self::TypeScript => "\
(call_expression function: (identifier) @ref)
(call_expression function: (member_expression property: (property_identifier) @ref))
(new_expression constructor: (identifier) @ref)
(type_identifier) @ref
",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// File walker
// ─────────────────────────────────────────────────────────────────────────────

/// Enumerate parseable source files under `root`, respecting `.gitignore`.
///
/// Returns `(path, language)` pairs. Files we don't have a grammar for are
/// silently skipped — the repo map only knows about languages it can parse.
fn walk_sources(root: &Path) -> Vec<(PathBuf, Language)> {
    let mut out = Vec::new();
    let walker = ignore::WalkBuilder::new(root).build();
    for entry in walker.flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        if let Some(lang) = Language::detect(path) {
            out.push((path.to_path_buf(), lang));
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Symbols and references extracted from one source file.
#[derive(Debug, Clone)]
pub struct ParsedFile {
    pub path: PathBuf,
    /// Top-level identifiers this file defines, in source order.
    pub definitions: Vec<String>,
    /// Identifiers this file uses but (likely) defines elsewhere. Deduplicated
    /// to a set because for graph construction we only care about presence,
    /// not multiplicity within a file.
    pub references: HashSet<String>,
}

fn parse_file(path: &Path, lang: Language) -> Result<ParsedFile> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let ts_lang = lang.tree_sitter_language();

    let mut parser = Parser::new();
    parser
        .set_language(&ts_lang)
        .context("setting tree-sitter language")?;
    let tree = parser
        .parse(&source, None)
        .context("tree-sitter returned no tree")?;

    let def_query =
        Query::new(&ts_lang, lang.definitions_query()).context("compiling definitions query")?;
    let ref_query =
        Query::new(&ts_lang, lang.references_query()).context("compiling references query")?;

    let definitions = run_query_named(&def_query, &tree, &source, "name");
    let references_vec = run_query_named(&ref_query, &tree, &source, "ref");

    Ok(ParsedFile {
        path: path.to_path_buf(),
        definitions,
        references: references_vec.into_iter().collect(),
    })
}

/// Run `query` against `tree`, returning the source text of every capture
/// whose name matches `capture_name`. Used for both definitions and
/// references with the same plumbing.
fn run_query_named(
    query: &Query,
    tree: &tree_sitter::Tree,
    source: &str,
    capture_name: &str,
) -> Vec<String> {
    let Some(want_idx) = query.capture_index_for_name(capture_name) else {
        return Vec::new();
    };
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), source.as_bytes());
    let mut out = Vec::new();
    while let Some(m) = matches.next() {
        for cap in m.captures {
            if cap.index == want_idx
                && let Ok(text) = cap.node.utf8_text(source.as_bytes())
            {
                out.push(text.to_string());
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph + PageRank
// ─────────────────────────────────────────────────────────────────────────────

/// PageRank damping factor — chance of following an edge vs. teleporting via
/// the personalization vector. 0.85 is the canonical value from the original
/// paper; values from 0.7 to 0.9 work fine for our purposes.
const DAMPING: f64 = 0.85;

/// Maximum iterations for PageRank convergence. In practice 30–50 is enough
/// for repos with a few thousand files; we cap higher to be safe.
const PAGERANK_MAX_ITER: usize = 100;

/// L1 convergence threshold — we stop iterating when the sum of absolute
/// rank changes falls below this. 1e-6 is sub-pixel for any reasonable repo.
const PAGERANK_TOL: f64 = 1e-6;

/// Compute PageRank on the file-reference graph.
///
/// `nodes` is the list of file indices; `out_edges[i]` is the list of
/// `(dst, weight)` tuples for outgoing edges from node `i`. `personalization`
/// is the teleport-target distribution — typically uniform, but elevated for
/// files in the user's `/add` set.
///
/// Returns one rank per node, summing to 1.0.
fn pagerank(
    n: usize,
    out_edges: &[Vec<(usize, f64)>],
    personalization: &[f64],
) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let p_sum: f64 = personalization.iter().sum();
    let p_sum = if p_sum > 0.0 { p_sum } else { n as f64 };
    let teleport: Vec<f64> = personalization
        .iter()
        .map(|p| (1.0 - DAMPING) * p / p_sum)
        .collect();

    // Precompute out-strength per node so we can normalize each iteration.
    let out_strength: Vec<f64> = out_edges
        .iter()
        .map(|edges| edges.iter().map(|(_, w)| *w).sum())
        .collect();

    let mut ranks = vec![1.0 / n as f64; n];
    let mut next = vec![0.0; n];

    for _ in 0..PAGERANK_MAX_ITER {
        // Distribute current rank along outgoing edges.
        next.fill(0.0);
        let mut dangling_mass = 0.0;
        for i in 0..n {
            if out_strength[i] == 0.0 {
                // Dangling node: redistributes its rank via the teleport vector.
                dangling_mass += ranks[i];
                continue;
            }
            for &(dst, w) in &out_edges[i] {
                next[dst] += DAMPING * ranks[i] * w / out_strength[i];
            }
        }
        let dangling_share = DAMPING * dangling_mass;
        for i in 0..n {
            next[i] += teleport[i] + dangling_share * personalization[i] / p_sum;
        }

        // Convergence: L1 distance from previous iteration.
        let delta: f64 = ranks
            .iter()
            .zip(&next)
            .map(|(a, b)| (a - b).abs())
            .sum();
        std::mem::swap(&mut ranks, &mut next);
        if delta < PAGERANK_TOL {
            break;
        }
    }
    ranks
}

// ─────────────────────────────────────────────────────────────────────────────
// RepoMap: the public face
// ─────────────────────────────────────────────────────────────────────────────

/// A built repo map: every parsed file plus the bookkeeping needed to
/// regenerate a rendered view with a different personalization vector
/// (i.e. a different `/add` set) without re-parsing.
pub struct RepoMap {
    root: PathBuf,
    files: Vec<ParsedFile>,
}

impl RepoMap {
    /// Walk and parse the repo. This is the expensive call (one-time cost,
    /// scales with codebase size). Parse failures on individual files are
    /// logged once via `tracing`-style eprintln and otherwise ignored.
    pub fn build(root: &Path) -> Self {
        let sources = walk_sources(root);
        let mut files = Vec::with_capacity(sources.len());
        for (path, lang) in sources {
            match parse_file(&path, lang) {
                Ok(p) => files.push(p),
                Err(e) => eprintln!("repomap: skipping {}: {e:#}", path.display()),
            }
        }
        Self {
            root: root.to_path_buf(),
            files,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Render the map as plain text suitable for sending to a model.
    ///
    /// `focused` is the set of canonical absolute paths the user has `/add`ed.
    /// Those files are *excluded* from the rendered map (the model sees their
    /// full contents elsewhere) but they elevate the personalization weight
    /// of their graph neighbors, so the map foregrounds related files.
    ///
    /// The output is truncated to roughly `token_budget` tokens by dropping
    /// the lowest-ranked files first.
    pub fn render(
        &self,
        focused: &HashSet<PathBuf>,
        token_budget: usize,
        tokenizer: &CoreBPE,
    ) -> String {
        if self.files.is_empty() {
            return String::new();
        }

        // Map every defined symbol to the set of files that define it. Used to
        // resolve references into edges below.
        let mut defs_to_files: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, f) in self.files.iter().enumerate() {
            for d in &f.definitions {
                defs_to_files.entry(d.as_str()).or_default().push(i);
            }
        }

        // Build outgoing edges: for each reference in file i, find which other
        // file(s) j define that symbol and add (j, +1) to out_edges[i].
        let n = self.files.len();
        let mut out_edges: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for (i, f) in self.files.iter().enumerate() {
            // Aggregate weight per destination so multiple references to the
            // same symbol don't bloat the edge list.
            let mut tmp: HashMap<usize, f64> = HashMap::new();
            for r in &f.references {
                if let Some(definers) = defs_to_files.get(r.as_str()) {
                    for &j in definers {
                        if j == i {
                            continue; // skip self-references
                        }
                        *tmp.entry(j).or_insert(0.0) += 1.0;
                    }
                }
            }
            out_edges[i] = tmp.into_iter().collect();
        }

        // Personalization: focused files get high weight, others uniform.
        let personalization: Vec<f64> = self
            .files
            .iter()
            .map(|f| {
                if focused.contains(&f.path) || canonicalize_match(&f.path, focused) {
                    100.0
                } else {
                    1.0
                }
            })
            .collect();

        let ranks = pagerank(n, &out_edges, &personalization);

        // Sort files by rank descending, then by path for ties.
        let mut order: Vec<usize> = (0..n)
            .filter(|&i| {
                // Exclude focused files (the model sees their full content already)
                !focused.contains(&self.files[i].path)
                    && !canonicalize_match(&self.files[i].path, focused)
            })
            .collect();
        order.sort_by(|&a, &b| {
            ranks[b]
                .partial_cmp(&ranks[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.files[a].path.cmp(&self.files[b].path))
        });

        // Build the rendered text by accumulating files until the token budget
        // is exceeded. We rank-order so the *most* relevant files come first.
        let header = "Repo map (PageRank-ranked symbols from files not in the chat):\n";
        let mut out = String::from(header);
        let mut tokens_so_far = tokenizer.encode_with_special_tokens(&out).len();

        for &i in &order {
            let f = &self.files[i];
            let entry = render_file_entry(&self.root, f);
            let entry_tokens = tokenizer.encode_with_special_tokens(&entry).len();
            if tokens_so_far + entry_tokens > token_budget {
                break;
            }
            out.push_str(&entry);
            tokens_so_far += entry_tokens;
        }

        out
    }
}

/// Compare a file's path against a focused set, accounting for canonical
/// vs. non-canonical paths.
///
/// `focused` typically contains canonicalized absolute paths (from FileContext),
/// but a ParsedFile's path is whatever the walker returned — usually relative
/// to the repo root. Try both forms before deciding.
fn canonicalize_match(path: &Path, focused: &HashSet<PathBuf>) -> bool {
    if let Ok(canon) = path.canonicalize() {
        if focused.contains(&canon) {
            return true;
        }
    }
    false
}

/// Render one file's entry in the map: relative path + a single-line list of
/// symbols. Compact-by-default to fit more files within the budget.
fn render_file_entry(root: &Path, f: &ParsedFile) -> String {
    let rel = f.path.strip_prefix(root).unwrap_or(&f.path);
    if f.definitions.is_empty() {
        return format!("{}\n", rel.display());
    }
    // Dedupe in case the query captured the same name twice (e.g. multiple
    // `impl` blocks for the same type).
    let mut seen = HashSet::new();
    let symbols: Vec<&str> = f
        .definitions
        .iter()
        .filter(|d| seen.insert(d.as_str()))
        .map(String::as_str)
        .collect();
    format!("{}:\n  {}\n", rel.display(), symbols.join(", "))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tempdir() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("yargent-repomap-test-{pid}-{n}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn language_detection_from_extension() {
        assert_eq!(Language::detect(Path::new("foo.rs")), Some(Language::Rust));
        assert_eq!(
            Language::detect(Path::new("foo.py")),
            Some(Language::Python)
        );
        assert_eq!(
            Language::detect(Path::new("foo.tsx")),
            Some(Language::TypeScript)
        );
        assert_eq!(Language::detect(Path::new("foo.md")), None);
        assert_eq!(Language::detect(Path::new("Makefile")), None);
    }

    #[test]
    fn parses_rust_definitions() {
        let dir = tempdir();
        let path = dir.join("lib.rs");
        std::fs::write(
            &path,
            "\
pub fn greet() {}
pub struct Greeter;
impl Greeter {
    pub fn hello(&self) {}
}
",
        )
        .unwrap();
        let parsed = parse_file(&path, Language::Rust).unwrap();
        assert!(parsed.definitions.iter().any(|d| d == "greet"));
        assert!(parsed.definitions.iter().any(|d| d == "Greeter"));
    }

    #[test]
    fn pagerank_uniform_gives_uniform_ranks() {
        // No edges, uniform personalization → all ranks equal to 1/n.
        let n = 4;
        let out_edges = vec![Vec::new(); n];
        let personalization = vec![1.0; n];
        let ranks = pagerank(n, &out_edges, &personalization);
        for r in &ranks {
            assert!((r - 0.25).abs() < 1e-3, "expected ~0.25, got {r}");
        }
    }

    #[test]
    fn pagerank_focused_node_dominates() {
        // 3 disconnected nodes, but node 0 is heavily favored by personalization.
        let n = 3;
        let out_edges = vec![Vec::new(); n];
        let personalization = vec![10.0, 1.0, 1.0];
        let ranks = pagerank(n, &out_edges, &personalization);
        assert!(ranks[0] > ranks[1] && ranks[0] > ranks[2]);
    }

    #[test]
    fn pagerank_flow_lifts_referenced_nodes() {
        // Two nodes referencing a single third node. Node 2 should win.
        //   0 ─┐
        //      ├─► 2
        //   1 ─┘
        let n = 3;
        let out_edges = vec![vec![(2, 1.0)], vec![(2, 1.0)], vec![]];
        let personalization = vec![1.0, 1.0, 1.0];
        let ranks = pagerank(n, &out_edges, &personalization);
        assert!(ranks[2] > ranks[0]);
        assert!(ranks[2] > ranks[1]);
    }

    #[test]
    fn render_skips_focused_files() {
        // Build a tiny repo, mark one file as focused, ensure it doesn't
        // appear in the rendered output.
        let dir = tempdir();
        let a = dir.join("a.rs");
        let b = dir.join("b.rs");
        std::fs::write(&a, "pub fn foo() {}\n").unwrap();
        std::fs::write(&b, "pub fn bar() {}\n").unwrap();

        let map = RepoMap::build(&dir);
        let mut focused = HashSet::new();
        focused.insert(a.canonicalize().unwrap());
        let tokenizer = tiktoken_rs::cl100k_base().unwrap();
        let rendered = map.render(&focused, 4096, &tokenizer);

        assert!(!rendered.contains("a.rs"));
        assert!(rendered.contains("b.rs"));
    }
}
