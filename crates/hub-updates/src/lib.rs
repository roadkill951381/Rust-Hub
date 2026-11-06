//! "Check for updates" for every piece Rust-Hub runs on.
//!
//! | Piece | Installed version from | Newest from | Upgrade |
//! |---|---|---|---|
//! | Rust-Hub | the commit it was built from, the git checkout | `git fetch` | `git pull`, `cargo build` |
//! | hub-gen packages | hub-gen's `/v1/versions` (or its venv) | PyPI, GitHub for git pins | `uv sync --upgrade-package` |
//! | Model weights | the Hugging Face cache | Hugging Face commits | `hub-gen pull` |
//! | ComfyUI | `comfyui_version.py` | GitHub releases | its update script, or `git pull` |
//! | ComfyUI custom nodes | their git checkouts | `git fetch` | `git pull` |
//! | ComfyUI model files | `ComfyUI/models` | (files have no versions) | download |
//! | Blender | `blender --version` | release tags on projects.blender.org | winget, or blender.org |
//! | uv, Rust, NVIDIA driver | their own commands | GitHub, `rustup check` | `uv self update`, `rustup update` |
//!
//! [`check`] only reads: it contacts those sites and runs version commands,
//! but changes nothing. Each [`Item`] that can move forward carries an
//! [`Upgrade`]: the exact commands, which [`launch`] runs in a window of
//! their own so the person sees what happens.

mod comfy;
mod hub_gen;
mod others;
mod run;
mod rust_hub;
mod version;
mod web;

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use run::{LaunchOptions, Launched, launch, run_inline};
pub use version::Version;
pub use web::Sources;

/// Where an item stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The newest version is installed.
    Current,
    /// A newer version can be installed.
    Available,
    /// A newer version exists, but Rust-Hub pins this one on purpose.
    HeldBack,
    /// Not installed or not downloaded yet.
    NotInstalled,
    /// Could not be checked; `detail` says why.
    Unknown,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Self::Current => "up to date",
            Self::Available => "update available",
            Self::HeldBack => "held back",
            Self::NotInstalled => "not installed",
            Self::Unknown => "not checked",
        }
    }
}

/// One command of an upgrade.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    /// Folder to run it in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<PathBuf>,
    pub program: String,
    pub args: Vec<String>,
}

impl Step {
    pub fn new<'a>(
        dir: Option<&Path>,
        program: &str,
        args: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        Self {
            dir: dir.map(Path::to_owned),
            program: program.to_owned(),
            args: args.into_iter().map(str::to_owned).collect(),
        }
    }
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&run::render(self))
    }
}

/// How to bring an item up to date.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Upgrade {
    /// Commands to run in order; empty when it is done by hand (see `link`).
    pub steps: Vec<Step>,
    /// What to know first, e.g. that ComfyUI must be closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// A page with the download or instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    /// The steps rebuild the hub, so it has to close while they run.
    #[serde(default)]
    pub rebuilds_hub: bool,
}

impl Upgrade {
    pub fn steps(steps: Vec<Step>) -> Self {
        Self {
            steps,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    #[must_use]
    pub fn link(mut self, link: impl Into<String>) -> Self {
        self.link = Some(link.into());
        self
    }

    /// The steps as lines to paste into a terminal.
    pub fn script(&self) -> String {
        let mut lines = Vec::new();
        let mut here: Option<&Path> = None;
        for step in &self.steps {
            if let Some(dir) = step.dir.as_deref().filter(|d| Some(*d) != here) {
                lines.push(format!("cd \"{}\"", dir.display()));
                here = Some(dir);
            }
            lines.push(step.to_string());
        }
        lines.join("\n")
    }
}

/// One thing that has a version: a program, a package, a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    /// Stable id, e.g. `hub-gen/transformers`, for `hub updates --apply`.
    pub id: String,
    pub name: String,
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installed: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latest: Option<String>,
    /// Why it is held back, what changed upstream, or why it was not checked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upgrade: Option<Upgrade>,
    /// Why the upgrade cannot run right now, e.g. ComfyUI is open.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked: Option<String>,
}

impl Item {
    pub fn new(id: impl Into<String>, name: impl Into<String>, status: Status) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            status,
            installed: None,
            latest: None,
            detail: None,
            upgrade: None,
            blocked: None,
        }
    }

    #[must_use]
    pub fn installed(mut self, v: impl Into<String>) -> Self {
        self.installed = Some(v.into());
        self
    }

    #[must_use]
    pub fn latest(mut self, v: impl Into<String>) -> Self {
        self.latest = Some(v.into());
        self
    }

    #[must_use]
    pub fn detail(mut self, d: impl Into<String>) -> Self {
        self.detail = Some(d.into());
        self
    }

    #[must_use]
    pub fn upgrade(mut self, u: Upgrade) -> Self {
        self.upgrade = Some(u);
        self
    }

    /// Worth showing by default: anything but up to date, leaving out
    /// things not installed that need nothing (weights downloaded on use).
    pub fn worth_showing(&self) -> bool {
        match self.status {
            Status::Current => false,
            Status::NotInstalled => self.upgrade.is_some(),
            _ => true,
        }
    }

    /// Whether it has commands that move it forward now.
    pub fn can_update(&self) -> bool {
        matches!(self.status, Status::Available | Status::NotInstalled)
            && self.blocked.is_none()
            && self.upgrade.as_ref().is_some_and(|u| !u.steps.is_empty())
    }
}

/// A group of items, e.g. ComfyUI with its nodes and model files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Piece {
    pub name: String,
    /// One line about the piece as a whole, e.g. "0.1.0, running".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    pub items: Vec<Item>,
}

/// Everything [`check`] found.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Report {
    pub pieces: Vec<Piece>,
}

impl Report {
    pub fn items(&self) -> impl Iterator<Item = &Item> {
        self.pieces.iter().flat_map(|p| &p.items)
    }

    pub fn find(&self, id: &str) -> Option<&Item> {
        self.items().find(|i| i.id == id)
    }

    /// Items with a newer version to install.
    pub fn available(&self) -> impl Iterator<Item = &Item> {
        self.items().filter(|i| i.status == Status::Available)
    }

    /// e.g. "3 updates available, 1 held back".
    pub fn summary(&self) -> String {
        let count = |s| self.items().filter(|i| i.status == s).count();
        let mut parts = vec![match count(Status::Available) {
            0 => "no updates found".to_owned(),
            1 => "1 update available".to_owned(),
            n => format!("{n} updates available"),
        }];
        for (status, what) in [
            (Status::HeldBack, "held back"),
            (Status::Unknown, "not checked"),
        ] {
            match count(status) {
                0 => {}
                n => parts.push(format!("{n} {what}")),
            }
        }
        parts.join(", ")
    }
}

/// What to check, and where.
#[derive(Debug, Clone)]
pub struct Setup {
    /// The Rust-Hub git checkout the hub was built from.
    pub source: Option<PathBuf>,
    /// The commit this build came from.
    pub built_commit: Option<String>,
    pub hub_gen: hub_generate::Client,
    /// `services/hub-gen`, to read its versions when the service is not running.
    pub hub_gen_dir: Option<PathBuf>,
    pub comfy: hub_comfy::Config,
    /// Look for Blender (it takes a few seconds to start).
    pub blender: bool,
    /// Check uv, Rust and the NVIDIA driver.
    pub tools: bool,
    pub sources: Sources,
}

impl Setup {
    /// The checkout this hub was built from (or `HUB_SOURCE`), hub-gen from
    /// `HUB_GEN_URL`, ComfyUI from `HUB_COMFY_*`, and the public sites.
    pub fn from_env() -> Self {
        let source = std::env::var_os("HUB_SOURCE")
            .map(PathBuf::from)
            .or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .canonicalize()
                    .ok()
            })
            .filter(|p| p.join("Cargo.toml").is_file());
        let hub_gen_dir = source
            .as_ref()
            .map(|s| s.join("services").join("hub-gen"))
            .filter(|d| d.join("pyproject.toml").is_file());
        Self {
            source,
            built_commit: option_env!("HUB_BUILT_COMMIT").map(str::to_owned),
            hub_gen: hub_generate::Client::from_env(),
            hub_gen_dir,
            comfy: hub_comfy::Config::from_env(),
            blender: true,
            tools: true,
            sources: Sources::default(),
        }
    }
}

/// Checks every piece, contacting the sites in parallel. Nothing is changed.
pub fn check(setup: &Setup) -> Report {
    let web = web::Web::new(setup.sources.clone());
    let web = &web;
    std::thread::scope(|s| {
        let mut tasks: Vec<std::thread::ScopedJoinHandle<'_, Piece>> = vec![
            s.spawn(move || rust_hub::check(setup)),
            s.spawn(move || hub_gen::check(setup, web)),
            s.spawn(move || comfy::check(&setup.comfy, web)),
        ];
        if setup.blender {
            tasks.push(s.spawn(move || others::blender(web)));
        }
        if setup.tools {
            tasks.push(s.spawn(move || others::tools(web)));
        }
        Report {
            pieces: tasks
                .into_iter()
                .map(|t| {
                    t.join().unwrap_or_else(|_| Piece {
                        name: "?".into(),
                        summary: Some("the check crashed".into()),
                        items: Vec::new(),
                    })
                })
                .collect(),
        }
    })
}

/// Runs `f` over `inputs` on up to `threads` threads, keeping their order.
fn parallel<T: Sync, R: Send>(inputs: &[T], threads: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    if inputs.is_empty() {
        return Vec::new();
    }
    let size = inputs.len().div_ceil(threads.max(1));
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = inputs
            .chunks(size)
            .map(|chunk| s.spawn(move || chunk.iter().map(f).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_default())
            .collect()
    })
}

/// The first `n` characters of a commit id.
fn short(commit: &str) -> &str {
    commit.get(..10).unwrap_or(commit)
}
