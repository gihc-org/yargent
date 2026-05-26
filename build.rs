use std::process::Command;

fn main() {
    // Capture short git SHA. Falls back to "unknown" if we're outside a
    // repo or if git isn't installed (e.g. in a tarball build or CI cache).
    let sha = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
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