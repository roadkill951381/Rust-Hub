//! Records the commit the hub was built from, so the update check can tell
//! when the checkout has moved on without a rebuild.

use std::path::Path;
use std::process::Command;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    if let Some(commit) = git(&["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=HUB_BUILT_COMMIT={commit}");
    }
    for path in ["HEAD", "refs/heads", "packed-refs"] {
        if let Some(p) = git(&["rev-parse", "--git-path", path]) {
            println!("cargo:rerun-if-changed={}", root.join(p).display());
        }
    }
}
