//! hub-gen's Python packages and the model weights in its Hugging Face cache.

use std::path::Path;
use std::time::Duration;

use hub_generate::{Package, Versions, Weights};

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
    let mut items = parallel(&versions.packages, 8, |p| {
        package(p, &versions, web, blocked.as_ref())
    });
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

fn package(p: &Package, versions: &Versions, web: &Web, blocked: Option<&String>) -> Item {
    let id = format!("hub-gen/{}", p.name.to_ascii_lowercase());
    let item = Item::new(id, p.name.clone(), Status::Unknown).installed(match &p.commit {
        Some(c) => format!("{} @ {}", p.installed, short(c)),
        None => p.installed.clone(),
    });
    if let Some(url) = &p.url {
        return git_pin(item, url, p.commit.as_deref(), web);
    }
    let all = match web.pypi_versions(&p.name) {
        Ok(v) => v,
        Err(e) => return item.detail(e),
    };
    let (any, allowed) = newest(all.iter().map(String::as_str), &p.wanted);
    let Some(installed) = Version::parse(&p.installed) else {
        return item.detail(format!("can't read the version {}", p.installed));
    };
    let mut item = item;
    match (allowed, any) {
        (Some((text, v)), any) if v > installed => {
            let mut args = sync_args(versions);
            args.extend(["--upgrade-package".to_owned(), p.name.clone()]);
            let step = Step {
                dir: Some(versions.project_dir.clone()),
                program: "uv".into(),
                args,
            };
            item = item.latest(text).upgrade(Upgrade::steps(vec![step]));
            item.status = Status::Available;
            item.blocked = blocked.cloned();
            if let Some((newer, _)) = any.filter(|(_, nv)| *nv > v) {
                item = item.detail(format!(
                    "{newer} is out, but pyproject.toml allows {}",
                    p.wanted
                ));
            }
        }
        (_, Some((text, v))) if v > installed => {
            item = item.latest(text).detail(format!(
                "pyproject.toml keeps it at {} because newer versions break hub-gen; see the comment there",
                p.wanted
            ));
            item.status = Status::HeldBack;
        }
        (_, Some((text, _))) => {
            item = item.latest(text);
            item.status = Status::Current;
        }
        (_, None) => item = item.detail("PyPI lists no releases"),
    }
    item
}

/// A package installed from a pinned git commit, e.g. hy3dgen.
fn git_pin(item: Item, url: &str, installed: Option<&str>, web: &Web) -> Item {
    let Some((repo, pinned)) = github_repo(url) else {
        return item.detail(format!("installed from {url}"));
    };
    let commit = installed.unwrap_or(pinned);
    match web.github_compare(&repo, commit, "main") {
        Ok(c) if c.ahead_by == 0 => {
            let mut item = item.latest(format!("{repo} main"));
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
    fn git_pins_name_their_repo() {
        assert_eq!(
            github_repo("git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2@f8db630"),
            Some(("Tencent-Hunyuan/Hunyuan3D-2".to_owned(), "f8db630"))
        );
        assert_eq!(github_repo("https://example.com/x.whl"), None);
    }
}
