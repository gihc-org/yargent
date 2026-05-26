use std::process::{Command, Stdio};

fn main() {
    // Capture short git SHA. Falls back to "unknown" if we're outside a
    // repo or if git isn't installed (e.g. in a tarball build or CI cache).
    // Stderr is silenced so the "fatal: not a git repository" message from
    // the fallback path doesn't pollute cargo's build output.
    let sha = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .stderr(Stdio::null())
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=YARGENT_GIT_SHA={sha}");

    // Re-run this build script whenever HEAD moves or a branch ref updates.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
}