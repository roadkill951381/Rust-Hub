//! Rust-Hub itself: the commit this build came from, against its checkout and
//! the checkout's remote.

use std::path::Path;
use std::time::Duration;

use crate::run::output;
use crate::{Item, Piece, Setup, Status, Step, Upgrade, short};

const REPO_URL: &str = "https://github.com/roadkill951381/Rust-Hub";

pub fn check(setup: &Setup) -> Piece {
    let version = env!("CARGO_PKG_VERSION");
    let built = setup.built_commit.as_deref();
    let installed = match built {
        Some(c) => format!("{version} ({})", short(c)),
        None => version.to_owned(),
    };
    let item = Item::new(
        "rust-hub",
        "Rust-Hub (hub, desktop app, MCP server)",
        Status::Unknown,
    )
    .installed(installed);
    let item = match &setup.source {
        Some(source) if source.join(".git").exists() => compare(item, source, built),
        _ => item
            .detail(format!(
                "this build's source folder is not on this computer, so it can't be compared; \
                 get the newest from {REPO_URL}"
            ))
            .upgrade(Upgrade::default().link(REPO_URL)),
    };
    Piece {
        name: "Rust-Hub".into(),
        summary: item.installed.clone(),
        items: vec![item],
    }
}

fn compare(item: Item, source: &Path, built: Option<&str>) -> Item {
    let git = |args: &[&str], secs: u64| {
        output(
            Path::new("git"),
            args,
            Some(source),
            Duration::from_secs(secs),
        )
    };
    let fetched = git(&["fetch", "--prune", "--quiet", "origin"], 60).is_some();
    let Some(head) = git(&["rev-parse", "HEAD"], 10) else {
        return item.detail(format!(
            "{} is not a working git checkout",
            source.display()
        ));
    };
    let branch = git(&["rev-parse", "--abbrev-ref", "HEAD"], 10).unwrap_or_default();
    let upstream = git(
        &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        10,
    );
    let main = git(&["rev-parse", "--verify", "--quiet", "origin/main"], 10)
        .map(|_| "origin/main".to_owned());
    let mut notes = Vec::new();
    if !fetched {
        notes.push(
            "could not reach the git remote, so this compares with what was last fetched"
                .to_owned(),
        );
    }
    let mut steps = Vec::new();
    let target = match (&upstream, &main) {
        (Some(up), _) => {
            steps.push(Step::new(Some(source), "git", ["pull", "--ff-only"]));
            Some(up.clone())
        }
        (None, Some(main)) => {
            notes.push(format!(
                "branch {branch} has no branch on the remote any more (merged or deleted), so this \
                 compares with main"
            ));
            steps.push(Step::new(Some(source), "git", ["switch", "main"]));
            steps.push(Step::new(Some(source), "git", ["pull", "--ff-only"]));
            Some(main.clone())
        }
        (None, None) => None,
    };
    let behind = target
        .as_deref()
        .and_then(|t| git(&["rev-list", "--count", &format!("HEAD..{t}")], 10))
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(0);
    let latest = target
        .as_deref()
        .and_then(|t| git(&["rev-parse", t], 10))
        .unwrap_or_else(|| head.clone());
    let stale_build = built.is_some_and(|b| b != head);
    if git(&["status", "--porcelain", "--untracked-files=no"], 10).is_some_and(|s| !s.is_empty()) {
        notes.push("the checkout has local changes; git pull stops if they clash".to_owned());
    }
    let build = Step::new(
        Some(source),
        "cargo",
        ["build", "--release", "-p", "hub-cli", "-p", "hub-app"],
    );
    let mut item = item.latest(format!(
        "{} ({})",
        target.as_deref().unwrap_or(&branch),
        short(&latest)
    ));
    if behind > 0 {
        notes.insert(
            0,
            format!(
                "{behind} new commit{} on {}",
                plural(behind),
                target.unwrap_or_default()
            ),
        );
        steps.push(build);
        item.status = Status::Available;
    } else if stale_build {
        notes.insert(
            0,
            "the checkout is newer than this build; rebuild to use it".to_owned(),
        );
        steps = vec![build];
        item.status = Status::Available;
    } else {
        item.status = Status::Current;
        steps.clear();
    }
    if !steps.is_empty() {
        item = item.upgrade(Upgrade {
            steps,
            note: Some(
                "The hub closes while it rebuilds, and opens again when the build is done.".into(),
            ),
            link: None,
            rebuilds_hub: true,
        });
    }
    if notes.is_empty() {
        item
    } else {
        item.detail(notes.join("; "))
    }
}

pub fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}
