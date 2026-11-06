//! hub-gen's Python packages and the model weights in its Hugging Face cache.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use hub_generate::{Package, Versions, Weights};

use crate::run::output;
use crate::rust_hub::plural;
use crate::version::{Version, newest};
use crate::web::Web;
use crate::{Item, Piece, Setup, Status, Step, Upgrade, parallel, short};

const NAME: &str = "Generation service (hub-gen)";

pub fn check(setup: &Setup, web: &Web) -> Piece {
    let (versions, running) = match setup.hub_gen.versions() {
        Ok(v) => (Ok(v), true),
        Err(e) => (offline(setup.hub_gen_dir.as_deref()).ok_or(e), false),
    };
    let versions = match versions {
        Ok(v) => v,
        Err(e) => return not_found(setup, &e),
    };
    let summary = format!(
        "{} {}{}",
        versions.version,
        if running {
            "running"
        } else {
            "not running (read from its folder)"
        },
        if versions.extras.is_empty() {
            String::new()
        } else {
            format!(", with {}", versions.extras.join(", "))
        }
    );
    let blocked =
        running.then(|| "hub-gen is running; stop it first, then start it again after".to_owned());
    let dir = versions.project_dir.as_path();
    let plan = uv_plan(dir);
    let mut items = parallel(&versions.packages, 8, |p| {
        package(p, &versions, web, blocked.as_ref(), plan.as_ref())
    });
    if let Some(plan) = &plan {
        items.extend(indirect(&versions, plan, blocked.as_ref()));
    }
    items.extend(parallel(&versions.weights, 4, |w| weights(w, dir, web)));
    Piece {
        name: NAME.into(),
        summary: Some(summary),
        items,
    }
}

/// `hub-gen versions --json` from the service's own venv.
fn offline(dir: Option<&Path>) -> Option<Versions> {
    let dir = dir?;
    let python = [
        dir.join(".venv").join("Scripts").join("python.exe"),
        dir.join(".venv").join("bin").join("python"),
    ]
    .into_iter()
    .find(|p| p.is_file())?;
    let text = crate::run::output(
        &python,
        &["-m", "hub_gen", "versions", "--json"],
        Some(dir),
        Duration::from_secs(60),
    )?;
    serde_json::from_str(&text).ok()
}

fn not_found(setup: &Setup, error: &str) -> Piece {
    let mut item = Item::new("hub-gen", "hub-gen", Status::Unknown).detail(error.to_owned());
    if let Some(dir) = &setup.hub_gen_dir
        && !dir.join(".venv").is_dir()
    {
        item = Item::new("hub-gen", "hub-gen", Status::NotInstalled)
            .detail("not installed yet; this installs it with the Hunyuan3D models")
            .upgrade(
                Upgrade::steps(vec![Step::new(
                    Some(dir),
                    "uv",
                    ["sync", "--extra", "hunyuan"],
                )])
                .link("https://docs.astral.sh/uv/getting-started/installation/")
                .note("Needs uv; the first run downloads PyTorch (a few GB)."),
            );
    }
    Piece {
        name: NAME.into(),
        summary: None,
        items: vec![item],
    }
}

fn sync_args(versions: &Versions) -> Vec<String> {
    let mut args = vec!["sync".to_owned()];
    for extra in &versions.extras {
        args.extend(["--extra".to_owned(), extra.clone()]);
    }
    args
}

/// Package updates uv's resolver would make, keeping every pin, every
/// package's own requirements and the CUDA index in mind. Normalised name ->
/// newest version it would lock.
pub type Plan = HashMap<String, String>;

/// Locks a copy of the service's project with `--upgrade` and compares it
/// with the current `uv.lock`, so nothing of the person's changes.
fn uv_plan(project: &Path) -> Option<Plan> {
    let lock = std::fs::read_to_string(project.join("uv.lock")).ok()?;
    let scratch = std::env::temp_dir()
        .join("rust-hub")
        .join(format!("uv-plan-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).ok()?;
    for file in ["pyproject.toml", "uv.lock", "README.md", ".python-version"] {
        if project.join(file).is_file() {
            std::fs::copy(project.join(file), scratch.join(file)).ok()?;
        }
    }
    let ran = output(
        Path::new("uv"),
        &["lock", "--upgrade", "--quiet"],
        Some(&scratch),
        Duration::from_secs(300),
    );
    let upgraded = std::fs::read_to_string(scratch.join("uv.lock")).ok();
    let _ = std::fs::remove_dir_all(&scratch);
    let upgraded = upgraded.filter(|u| *u != lock || ran.is_some())?;
    Some(diff_locks(&lock, &upgraded))
}

/// Each package's newest version in a `uv.lock`.
fn lock_versions(lock: &str) -> HashMap<String, Version> {
    let mut out: HashMap<String, Version> = HashMap::new();
    let mut name: Option<String> = None;
    for line in lock.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            name = None;
        } else if let Some(v) = line.strip_prefix("name = ") {
            name = Some(normalise(v.trim_matches('"')));
        } else if let Some(v) = line.strip_prefix("version = ")
            && let (Some(n), Some(v)) = (name.take(), Version::parse(v.trim_matches('"')))
        {
            let entry = out.entry(n).or_insert_with(|| v.clone());
            if v > *entry {
                *entry = v;
            }
        }
    }
    out
}

fn diff_locks(before: &str, after: &str) -> Plan {
    let before = lock_versions(before);
    let mut texts: HashMap<String, String> = HashMap::new();
    // Keep the version as written (with `+cu128`) for display.
    let mut name: Option<String> = None;
    for line in after.lines().map(str::trim) {
        if let Some(v) = line.strip_prefix("name = ") {
            name = Some(normalise(v.trim_matches('"')));
        } else if let Some(v) = line.strip_prefix("version = ")
            && let Some(n) = name.take()
        {
            let v = v.trim_matches('"').to_owned();
            // Prefer a CUDA build's name (`+cu128`) over its plain twin.
            let newer = texts.get(&n).is_none_or(|old| {
                let (a, b) = (Version::parse(&v), Version::parse(old));
                a > b || (a == b && v.contains('+') && !old.contains('+'))
            });
            if newer {
                texts.insert(n, v);
            }
        }
    }
    texts
        .into_iter()
        .filter(|(n, v)| {
            let (Some(old), Some(new)) = (before.get(n), Version::parse(v)) else {
                return false;
            };
            new > *old
        })
        .collect()
}

fn normalise(name: &str) -> String {
    name.to_ascii_lowercase().replace(['_', '.'], "-")
}

fn upgrade_step(versions: &Versions, package: Option<&str>) -> Upgrade {
    let mut args = sync_args(versions);
    match package {
        Some(name) => args.extend(["--upgrade-package".to_owned(), name.to_owned()]),
        None => args.push("--upgrade".to_owned()),
    }
    Upgrade::steps(vec![Step {
        dir: Some(versions.project_dir.clone()),
        program: "uv".into(),
        args,
    }])
}

fn package(
    p: &Package,
    versions: &Versions,
    web: &Web,
    blocked: Option<&String>,
    plan: Option<&Plan>,
) -> Item {
    let id = format!("hub-gen/{}", p.name.to_ascii_lowercase());
    let mut item = Item::new(id, p.name.clone(), Status::Unknown).installed(match &p.commit {
        Some(c) => format!("{} @ {}", p.installed, short(c)),
        None => p.installed.clone(),
    });
    if let Some(url) = &p.url {
        return git_pin(item, url, p.commit.as_deref(), web);
    }
    // A local build such as torch's `+cu128` comes from its own index, so
    // PyPI's newest release says nothing about it.
    let local_build = p.installed.contains('+');
    if let Some(to) = plan.and_then(|plan| plan.get(&normalise(&p.name))) {
        item = item
            .latest(to.clone())
            .upgrade(upgrade_step(versions, Some(&p.name)));
        item.status = Status::Available;
        item.blocked = blocked.cloned();
        return item;
    }
    let Some(installed) = Version::parse(&p.installed) else {
        return item.detail(format!("can't read the version {}", p.installed));
    };
    if local_build {
        if plan.is_some() {
            item.status = Status::Current;
            item.latest.clone_from(&item.installed);
            return item.detail("a CUDA build; uv finds no newer one on its CUDA index");
        }
        return item
            .detail("a CUDA build, which only uv on the computer running hub-gen can compare");
    }
    let all = match web.pypi_versions(&p.name) {
        Ok(v) => v,
        Err(e) => return item.detail(e),
    };
    let (any, allowed) = newest(all.iter().map(String::as_str), &p.wanted);
    match (plan, allowed, any) {
        // uv's answer: no move possible, whatever PyPI has.
        (Some(_), _, Some((text, v))) if v > installed => {
            item = item.latest(text).detail(if allows_any(&p.wanted, &v) {
                "uv keeps it here: another package hub-gen uses doesn't accept newer releases \
                 yet (with the transformers pin, usually diffusers or a package it needs)"
                    .to_owned()
            } else {
                format!(
                    "pyproject.toml keeps it at {}, because newer versions break hub-gen; \
                     see the comment there",
                    p.wanted
                )
            });
            item.status = Status::HeldBack;
        }
        // No uv here (hub-gen runs elsewhere): judge by the range alone.
        (None, Some((text, v)), _) if v > installed => {
            item = item
                .latest(text)
                .upgrade(upgrade_step(versions, Some(&p.name)));
            item.status = Status::Available;
            item.blocked = blocked.cloned();
        }
        (None, _, Some((text, v))) if v > installed => {
            item = item.latest(text).detail(format!(
                "pyproject.toml keeps it at {}, because newer versions break hub-gen",
                p.wanted
            ));
            item.status = Status::HeldBack;
        }
        (_, _, Some((text, _))) => {
            item = item.latest(text);
            item.status = Status::Current;
        }
        (_, _, None) => item = item.detail("PyPI lists no releases"),
    }
    item
}

fn allows_any(range: &str, v: &Version) -> bool {
    crate::version::allows(range, v)
}

/// The packages hub-gen pulls in indirectly (huggingface-hub and so on).
fn indirect(versions: &Versions, plan: &Plan, blocked: Option<&String>) -> Option<Item> {
    let direct: Vec<_> = versions
        .packages
        .iter()
        .map(|p| normalise(&p.name))
        .collect();
    let mut others: Vec<_> = plan
        .iter()
        .filter(|(name, _)| !direct.contains(name))
        .map(|(name, to)| format!("{name} {to}"))
        .collect();
    if others.is_empty() {
        return None;
    }
    others.sort();
    let n = others.len();
    let shown = others
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let mut item = Item::new(
        "hub-gen/all",
        "Everything at once, with the packages hub-gen's packages need",
        Status::Available,
    )
    .latest(format!("{n} indirect update{}", plural(n as u64)))
    .detail(format!("{shown}{}", if n > 8 { ", ..." } else { "" }))
    .upgrade(upgrade_step(versions, None));
    item.blocked = blocked.cloned();
    Some(item)
}

/// A package installed from a pinned git commit, e.g. hy3dgen.
fn git_pin(item: Item, url: &str, installed: Option<&str>, web: &Web) -> Item {
    let Some((repo, pinned)) = github_repo(url) else {
        return item.detail(format!("installed from {url}"));
    };
    let commit = installed.unwrap_or(pinned);
    match web.github_compare(&repo, commit, "main") {
        Ok(c) if c.ahead_by == 0 => {
            let mut item = item
                .latest(format!("{repo} main = {}", short(commit)))
                .detail("pinned in pyproject.toml; main has no newer commits");
            item.status = Status::Current;
            item
        }
        Ok(c) => {
            let mut item = item
                .latest(format!("{} newer commit{} on {repo}", c.ahead_by, plural(c.ahead_by)))
                .detail(format!(
                    "pinned to a tested commit in pyproject.toml; hub-gen's adapter must be checked \
                     against newer code before the pin moves: https://github.com/{repo}/compare/{}...main",
                    short(commit)
                ));
            item.status = Status::HeldBack;
            item
        }
        Err(e) => item.detail(e),
    }
}

/// `owner/name` and the commit from `git+https://github.com/owner/name@commit`.
fn github_repo(url: &str) -> Option<(String, &str)> {
    let rest = url.strip_prefix("git+")?.split("github.com/").nth(1)?;
    let (repo, commit) = rest.split_once('@')?;
    Some((repo.trim_end_matches(".git").to_owned(), commit))
}

fn weights(w: &Weights, project: &Path, web: &Web) -> Item {
    let id = format!("weights/{}", w.repo.to_ascii_lowercase());
    let name = format!("{} ({})", w.repo, w.models.join(", "));
    let pull = |what: &str| {
        Upgrade::steps(vec![Step::new(
            Some(project),
            "uv",
            ["run", "--no-sync", "hub-gen", "pull", what],
        )])
        .note("hub-gen keeps using the weights it has loaded until it restarts.")
    };
    let Some(revision) = &w.revision else {
        return Item::new(id, name, Status::NotInstalled).detail(
            "not downloaded; hub-gen downloads it the first time one of these models is used",
        );
    };
    let item = Item::new(id, name, Status::Unknown).installed(short(revision));
    let commits = match web.hf_commits(&w.repo) {
        Ok(c) => c,
        Err(e) => return item.detail(e),
    };
    let Some(latest) = commits.first() else {
        return item.detail("Hugging Face lists no commits");
    };
    let mut item = item.latest(format!(
        "{} ({})",
        short(&latest.id),
        latest.date.get(..10).unwrap_or("")
    ));
    if latest.id == *revision {
        item.status = Status::Current;
    } else {
        let newer: Vec<_> = commits.iter().take_while(|c| c.id != *revision).collect();
        let count = if newer.len() == commits.len() {
            format!("more than {}", newer.len())
        } else {
            newer.len().to_string()
        };
        let titles: Vec<_> = newer
            .iter()
            .take(3)
            .map(|c| format!("\"{}\"", c.title))
            .collect();
        item = item
            .detail(format!(
                "{count} newer commit{} on Hugging Face: {}{}",
                plural(newer.len() as u64),
                titles.join(", "),
                if newer.len() > 3 { ", ..." } else { "" }
            ))
            .upgrade(pull(&w.repo));
        item.status = Status::Available;
    }
    if let Some(copy) = &w.copy {
        item.detail = Some(format!(
            "{}; hy3dgen loads {copy} instead of the cache",
            item.detail.as_deref().unwrap_or("")
        ));
    }
    item
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_files_are_compared() {
        let before = r#"
[[package]]
name = "torch"
version = "2.11.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "torch"
version = "2.11.0+cu128"
source = { registry = "https://download.pytorch.org/whl/cu128" }

[[package]]
name = "huggingface-hub"
version = "0.36.2"
dependencies = [
    { name = "filelock" },
]

[[package]]
name = "pillow"
version = "12.3.0"
"#;
        let after = before
            .replace("2.11.0+cu128", "2.15.0+cu128")
            .replace("\"2.11.0\"", "\"2.15.0\"")
            .replace("0.36.2", "0.37.0")
            .replace("name = \"huggingface-hub\"", "name = \"huggingface_hub\"");
        let plan = diff_locks(before, &after);
        assert_eq!(plan.len(), 2, "{plan:?}");
        assert_eq!(plan["torch"], "2.15.0+cu128");
        assert_eq!(plan["huggingface-hub"], "0.37.0");
        assert!(diff_locks(before, before).is_empty());
    }

    #[test]
    fn git_pins_name_their_repo() {
        assert_eq!(
            github_repo("git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2@f8db630"),
            Some(("Tencent-Hunyuan/Hunyuan3D-2".to_owned(), "f8db630"))
        );
        assert_eq!(github_repo("https://example.com/x.whl"), None);
    }
}
