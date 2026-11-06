//! ComfyUI, its custom nodes, and the model files the built-in workflows load.

use std::path::Path;
use std::time::Duration;

use hub_comfy::{Config, Installation, MODEL_FILES, ModelFile};

use crate::run::output;
use crate::rust_hub::plural;
use crate::version::Version;
use crate::web::Web;
use crate::{Item, Piece, Status, Step, Upgrade, short};

const NAME: &str = "ComfyUI";
const RELEASES: &str = "comfyanonymous/ComfyUI";

pub fn check(config: &Config, web: &Web) -> Piece {
    let Some(install) = config.installation() else {
        let item = Item::new("comfyui", "ComfyUI", Status::NotInstalled)
            .detail(match &config.dir {
                Some(d) => format!("no ComfyUI found in {}", d.display()),
                None => "not found; set HUB_COMFY_DIR if it is somewhere unusual".to_owned(),
            })
            .upgrade(
                Upgrade::default().link("https://github.com/comfyanonymous/ComfyUI#installing"),
            );
        return Piece {
            name: NAME.into(),
            summary: Some("optional: Rust-Hub can use it instead of hub-gen".into()),
            items: vec![item],
        };
    };
    let running = hub_comfy::ComfyUi::new(config.clone()).is_running();
    let blocked = running.then(|| "ComfyUI is running; close its window first".to_owned());
    let mut items = vec![core(&install, web, blocked.clone())];
    items.extend(custom_nodes(&install, blocked.as_ref()));
    let models = install.app.join("models");
    items.extend(MODEL_FILES.iter().map(|f| model_file(f, &models)));
    Piece {
        name: NAME.into(),
        summary: Some(format!(
            "{}{}",
            install
                .portable
                .as_deref()
                .unwrap_or(&install.app)
                .display(),
            if running { ", running" } else { "" }
        )),
        items,
    }
}

/// `__version__ = "0.3.43"` from `comfyui_version.py`.
fn installed_version(app: &Path) -> Option<String> {
    let text = std::fs::read_to_string(app.join("comfyui_version.py")).ok()?;
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("__version__"))?;
    Some(line.split(['"', '\'']).nth(1)?.to_owned())
}

fn core(install: &Installation, web: &Web, blocked: Option<String>) -> Item {
    let mut item = Item::new("comfyui", "ComfyUI", Status::Unknown);
    item.blocked = blocked;
    let installed = installed_version(&install.app);
    if let Some(v) = &installed {
        item = item.installed(v.clone());
    }
    let latest = match web.github_release(RELEASES) {
        Ok(tag) => tag,
        Err(e) => return item.detail(e),
    };
    item = item.latest(latest.trim_start_matches('v').to_owned());
    let (Some(have), Some(newest)) = (
        installed.as_deref().and_then(Version::parse),
        Version::parse(&latest),
    ) else {
        return item.detail("can't read the installed version (comfyui_version.py)");
    };
    if have >= newest {
        item.status = Status::Current;
        return item;
    }
    item.status = Status::Available;
    let upgrade = if let Some(portable) = &install.portable {
        let update = portable.join("update");
        let script = ["update_comfyui_stable.bat", "update_comfyui.bat"]
            .into_iter()
            .find(|s| update.join(s).is_file());
        match script {
            Some(script) => Upgrade::steps(vec![Step::new(Some(&update), script, [])])
                .note("ComfyUI's own updater; it waits for a key press at the end."),
            None => Upgrade::default().link("https://github.com/comfyanonymous/ComfyUI/releases"),
        }
    } else {
        let python = install.python.display().to_string();
        Upgrade::steps(vec![
            Step::new(Some(&install.app), "git", ["pull", "--ff-only"]),
            Step::new(
                Some(&install.app),
                &python,
                ["-m", "pip", "install", "-r", "requirements.txt"],
            ),
        ])
    };
    item.upgrade(upgrade)
}

fn custom_nodes(install: &Installation, blocked: Option<&String>) -> Vec<Item> {
    let Ok(entries) = std::fs::read_dir(install.app.join("custom_nodes")) else {
        return Vec::new();
    };
    let mut dirs: Vec<_> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            !name.starts_with('.') && !name.starts_with("__") && !name.ends_with(".disabled")
        })
        .collect();
    dirs.sort();
    crate::parallel(&dirs, 4, |dir| node(dir, install, blocked))
}

fn node(dir: &Path, install: &Installation, blocked: Option<&String>) -> Item {
    let name = dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let mut item = Item::new(
        format!("comfyui/{}", name.to_ascii_lowercase()),
        format!("{name} (custom node)"),
        Status::Unknown,
    );
    if !dir.join(".git").exists() {
        if let Some(v) = pyproject_version(dir) {
            item = item.installed(v);
        }
        return item.detail(
            "not a git checkout, so it can't be checked here; update it in ComfyUI-Manager, \
             or reinstall it with git clone",
        );
    }
    let git =
        |args: &[&str], secs| output(Path::new("git"), args, Some(dir), Duration::from_secs(secs));
    let fetched = git(&["fetch", "--quiet"], 60).is_some();
    let Some(head) = git(&["rev-parse", "HEAD"], 10) else {
        return item.detail("its git checkout can't be read");
    };
    item = item.installed(short(&head));
    let Some(upstream) = git(&["rev-parse", "@{u}"], 10) else {
        return item.detail("its checkout follows no remote branch");
    };
    item = item.latest(short(&upstream));
    let behind = git(&["rev-list", "--count", "HEAD..@{u}"], 10)
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(0);
    if behind == 0 {
        item.status = Status::Current;
        return if fetched {
            item
        } else {
            item.detail("could not reach its remote")
        };
    }
    let mut steps = vec![Step::new(Some(dir), "git", ["pull", "--ff-only"])];
    if dir.join("requirements.txt").is_file() {
        let python = install.python.display().to_string();
        let mut args = vec!["-m", "pip", "install", "-r", "requirements.txt"];
        if install.portable.is_some() {
            args.insert(0, "-s");
        }
        steps.push(Step::new(Some(dir), &python, args));
    }
    item.status = Status::Available;
    item.blocked = blocked.cloned();
    item.detail(format!("{behind} new commit{}", plural(behind)))
        .upgrade(Upgrade::steps(steps))
}

/// `version = "1.2.3"` from a node pack's `pyproject.toml`.
fn pyproject_version(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("pyproject.toml")).ok()?;
    let line = text
        .lines()
        .find(|l| l.trim_start().starts_with("version") && l.contains('='))?;
    Some(line.split(['"', '\'']).nth(1)?.to_owned())
}

fn model_file(file: &ModelFile, models: &Path) -> Item {
    let id = format!("comfyui-model/{}", file.name.to_ascii_lowercase());
    let name = format!("{} (for {})", file.name, file.workflows.join(", "));
    let found = file
        .folders
        .iter()
        .map(|f| models.join(f).join(file.name))
        .find(|p| p.is_file());
    let page = format!("https://huggingface.co/{}", file.repo);
    if let Some(path) = found {
        let size = std::fs::metadata(&path).map_or(0, |m| m.len());
        return Item::new(id, name, Status::Current)
            .installed(format!("{:.1} GB", size as f64 / 1e9))
            .detail(format!(
                "a fixed release with no version number; in {}",
                path.parent().unwrap_or(models).display()
            ));
    }
    let dest = models.join(file.folders[0]);
    let mut item = Item::new(id, name, Status::NotInstalled).detail(format!(
        "only needed for the {} workflow{}; goes in {}",
        file.workflows.join(" and "),
        if file.workflows.len() == 1 { "" } else { "s" },
        dest.display()
    ));
    let upgrade = if file.gated {
        Upgrade::default()
            .link(page)
            .note("Sign in to Hugging Face and accept the licence on that page, then download it into the folder above.")
    } else {
        let url = format!("{page}/resolve/main/{}", file.name);
        let out = dest.join(file.name).display().to_string();
        Upgrade::steps(vec![Step::new(
            Some(models),
            "curl",
            ["-L", "--fail", "--create-dirs", "-o", &out, &url],
        )])
        .link(page)
    };
    item = item.upgrade(upgrade);
    item
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_are_read_from_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("comfyui_version.py"),
            "# generated\n__version__ = \"0.42.1\"\n",
        )
        .unwrap();
        assert_eq!(installed_version(dir.path()).as_deref(), Some("0.42.1"));
        std::fs::write(
            dir.path().join("pyproject.toml"),
            "[project]\nname = \"ComfyUI-GGUF\"\nversion = \"1.1.4\"\n",
        )
        .unwrap();
        assert_eq!(pyproject_version(dir.path()).as_deref(), Some("1.1.4"));
    }
}
