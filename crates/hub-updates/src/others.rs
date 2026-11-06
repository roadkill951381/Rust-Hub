//! Blender, and the tools the hub is built and run with.

use std::path::Path;
use std::time::Duration;

use crate::run::{output, output_any_exit};
use crate::version::Version;
use crate::web::Web;
use crate::{Item, Piece, Status, Step, Upgrade};

const BLENDER_PAGE: &str = "https://www.blender.org/download/";

pub fn blender(web: &Web) -> Piece {
    let item = Item::new("blender", "Blender", Status::Unknown);
    let found = hub_blender::Blender::locate();
    let winget = |verb: &str| {
        Upgrade::steps(if cfg!(windows) {
            vec![Step::new(
                None,
                "winget",
                [verb, "--id", "BlenderFoundation.Blender", "--exact"],
            )]
        } else {
            Vec::new()
        })
        .link(BLENDER_PAGE)
    };
    let item = match &found {
        Ok(b) => item.installed(b.version().to_owned()),
        Err(e) => {
            let mut item = item.detail(e.clone()).upgrade(winget("install"));
            item.status = Status::NotInstalled;
            return piece("Blender", None, vec![item]);
        }
    };
    let summary = found
        .as_ref()
        .ok()
        .map(|b| b.program().display().to_string());
    let latest = match web.blender_latest() {
        Ok(v) => v,
        Err(e) => return piece("Blender", summary, vec![item.detail(e)]),
    };
    let mut item = item.latest(latest.clone());
    let installed = item.installed.as_deref().and_then(Version::parse);
    match (installed, Version::parse(&latest)) {
        (Some(have), Some(newest)) if have >= newest => item.status = Status::Current,
        (Some(_), Some(_)) => {
            item.status = Status::Available;
            item =
                item.upgrade(winget("upgrade").note(
                    "Rust-Hub uses the newest Blender it finds, or the one HUB_BLENDER names.",
                ));
        }
        _ => item = item.detail("can't compare the version numbers"),
    }
    piece("Blender", summary, vec![item])
}

pub fn tools(web: &Web) -> Piece {
    let items = std::thread::scope(|s| {
        let uv = s.spawn(|| uv(web));
        let rust = s.spawn(rust);
        let driver = s.spawn(nvidia);
        [uv, rust, driver]
            .into_iter()
            .filter_map(|h| h.join().ok().flatten())
            .collect()
    });
    piece("Tools", None, items)
}

fn piece(name: &str, summary: Option<String>, items: Vec<Item>) -> Piece {
    Piece {
        name: name.into(),
        summary,
        items,
    }
}

fn run(program: &str, args: &[&str], secs: u64) -> Option<String> {
    output(Path::new(program), args, None, Duration::from_secs(secs))
}

/// `uv 0.13.7 (abc123 2026-10-30)` against uv's latest GitHub release.
fn uv(web: &Web) -> Option<Item> {
    let text = run("uv", &["--version"], 10)?;
    let installed = text.split_whitespace().nth(1)?.to_owned();
    let mut item =
        Item::new("uv", "uv (runs hub-gen's Python)", Status::Unknown).installed(installed.clone());
    match web.github_release("astral-sh/uv") {
        Ok(tag) => {
            item = item.latest(tag.clone());
            if Version::parse(&installed) >= Version::parse(&tag) {
                item.status = Status::Current;
            } else {
                item.status = Status::Available;
                item = item.upgrade(
                    Upgrade::steps(vec![Step::new(None, "uv", ["self", "update"])])
                        .note("If uv came from winget, use: winget upgrade astral-sh.uv"),
                );
            }
        }
        Err(e) => item = item.detail(e),
    }
    Some(item)
}

/// `rustup check`, which asks the Rust release server itself.
fn rust() -> Option<Item> {
    let rustc = run("rustc", &["--version"], 10)?;
    let installed = rustc.split_whitespace().nth(1)?.to_owned();
    let mut item = Item::new("rust", "Rust (builds the hub)", Status::Unknown).installed(installed);
    let Some(report) = output_any_exit(
        Path::new("rustup"),
        &["check"],
        None,
        Duration::from_secs(60),
    ) else {
        return Some(item.detail("rustup check failed or isn't installed"));
    };
    let line = report
        .lines()
        .find(|l| l.starts_with("stable"))
        .map(str::to_ascii_lowercase);
    match line {
        Some(l) if l.contains("update available") => {
            let latest = l
                .rsplit("-> ")
                .next()
                .and_then(|r| r.split_whitespace().next());
            if let Some(v) = latest {
                item = item.latest(v.to_owned());
            }
            item.status = Status::Available;
            item = item.upgrade(Upgrade::steps(vec![Step::new(
                None,
                "rustup",
                ["update", "stable"],
            )]));
        }
        Some(l) if l.contains("up to date") => {
            item.latest.clone_from(&item.installed);
            item.status = Status::Current;
        }
        _ => item = item.detail("rustup reported no stable toolchain"),
    }
    Some(item)
}

fn nvidia() -> Option<Item> {
    let text = run(
        "nvidia-smi",
        &["--query-gpu=driver_version,name", "--format=csv,noheader"],
        15,
    )?;
    let first = text.lines().next()?;
    let (version, gpu) = first.split_once(',').unwrap_or((first, ""));
    Some(
        Item::new(
            "nvidia-driver",
            format!("NVIDIA driver ({})", gpu.trim()),
            Status::Unknown,
        )
        .installed(version.trim().to_owned())
        .detail(
            "NVIDIA publishes no version feed to compare with; the NVIDIA app checks for drivers",
        )
        .upgrade(Upgrade::default().link("https://www.nvidia.com/en-us/software/nvidia-app/")),
    )
}
