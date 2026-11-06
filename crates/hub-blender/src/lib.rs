//! Blender as a mesh engine for Rust-Hub.
//!
//! Each operation runs Blender in the background on a temporary GLB:
//! write the mesh, run [`hub_jobs.py`](../src/hub_jobs.py) with a JSON job,
//! read the result back. Blender is found through `HUB_BLENDER`, the `PATH`,
//! or its standard install folders. `HUB_BLENDER` may also point to a Python
//! interpreter with Blender's `bpy` module installed, which CI uses.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use hub_core::engine::{MeshOp, MeshTools};
use hub_core::scene::{Material, Object, ObjectId, Transform};
use hub_core::{Mesh, io};
use serde::Deserialize;
use serde_json::json;

const SCRIPT: &str = include_str!("hub_jobs.py");

/// Oldest Blender release the jobs are tested against.
pub const MIN_VERSION: (u32, u32) = (4, 2);

/// A located, working Blender installation.
#[derive(Debug, Clone)]
pub struct Blender {
    program: PathBuf,
    /// True when `program` is a Python with the `bpy` module rather than Blender.
    python_module: bool,
    version: String,
}

#[derive(Debug, Deserialize)]
struct Report {
    ok: bool,
    blender: String,
    #[serde(default)]
    error: Option<String>,
}

impl Blender {
    /// Finds Blender: `HUB_BLENDER`, then `blender` on the `PATH`, then the
    /// standard install folders for this platform (newest version first).
    pub fn locate() -> Result<Self, String> {
        let mut tried = Vec::new();
        for candidate in candidates() {
            match Self::at(&candidate) {
                Ok(blender) => return Ok(blender),
                Err(e) => tried.push(format!("{}: {e}", candidate.display())),
            }
        }
        Err(if tried.is_empty() {
            "Blender was not found. Install Blender 5.1 or newer from blender.org, \
             or set HUB_BLENDER to the blender executable"
                .to_owned()
        } else {
            format!("no working Blender found ({})", tried.join("; "))
        })
    }

    /// Uses the Blender (or Python with `bpy`) at `program`, checking that it runs.
    pub fn at(program: &Path) -> Result<Self, String> {
        let name = program
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_lowercase();
        let mut blender = Self {
            program: program.to_owned(),
            python_module: name.starts_with("python"),
            version: String::new(),
        };
        let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let report = blender.run(dir.path(), "version", None, &json!({}))?;
        let (major, minor) = parse_version(&report.blender);
        if (major, minor) < MIN_VERSION {
            return Err(format!(
                "Blender {} is too old; install {}.{} or newer",
                report.blender, MIN_VERSION.0, MIN_VERSION.1
            ));
        }
        blender.version = report.blender;
        Ok(blender)
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn program(&self) -> &Path {
        &self.program
    }

    /// Shares this Blender as a [`MeshTools`] engine.
    pub fn into_engine(self) -> Arc<dyn MeshTools> {
        Arc::new(self)
    }

    /// Runs one job in `dir` and returns Blender's report.
    fn run(
        &self,
        dir: &Path,
        op: &str,
        input: Option<&Mesh>,
        params: &serde_json::Value,
    ) -> Result<Report, String> {
        let script = dir.join("hub_jobs.py");
        let job = dir.join("job.json");
        let report = dir.join("report.json");
        let input_path = dir.join("input.glb");
        let output_path = dir.join("output.glb");
        let write = |path: &Path, bytes: &[u8]| {
            std::fs::write(path, bytes)
                .map_err(|e| format!("could not write {}: {e}", path.display()))
        };
        write(&script, SCRIPT.as_bytes())?;
        if let Some(mesh) = input {
            let object = Object {
                id: ObjectId(0),
                name: "input".into(),
                mesh: Arc::new(mesh.clone()),
                transform: Transform::IDENTITY,
                material: Material::default(),
            };
            io::export(&input_path, &[&object]).map_err(|e| e.to_string())?;
        }
        let spec = json!({
            "op": op,
            "input": input_path,
            "output": output_path,
            "report": report,
            "params": params,
        });
        write(&job, spec.to_string().as_bytes())?;

        let mut args: Vec<OsString> = Vec::new();
        if !self.python_module {
            args.extend(
                [
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "3",
                    "--python",
                ]
                .map(OsString::from),
            );
        }
        args.extend([script.into_os_string(), "--".into(), job.into_os_string()]);
        tracing::debug!(program = %self.program.display(), op, "running Blender job");
        let output = Command::new(&self.program)
            .args(&args)
            .current_dir(dir)
            .output()
            .map_err(|e| format!("could not start {}: {e}", self.program.display()))?;

        let text = std::fs::read_to_string(&report).map_err(|_| {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: Vec<&str> = stderr.lines().rev().take(8).collect();
            format!(
                "Blender exited with {} and no report; last output: {}",
                output.status,
                tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
            )
        })?;
        let parsed: Report =
            serde_json::from_str(&text).map_err(|e| format!("unreadable Blender report: {e}"))?;
        if parsed.ok {
            Ok(parsed)
        } else {
            Err(format!(
                "Blender {} failed: {}",
                parsed.blender,
                parsed.error.as_deref().unwrap_or("unknown error")
            ))
        }
    }

    /// Runs `op` on `mesh` and returns the resulting mesh, everything joined.
    fn process(&self, op: &str, mesh: &Mesh, params: &serde_json::Value) -> Result<Mesh, String> {
        let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        self.run(dir.path(), op, Some(mesh), params)?;
        let objects = io::import(&dir.path().join("output.glb")).map_err(|e| e.to_string())?;
        let mut out = Mesh::default();
        for o in objects {
            out.append(&o.mesh.transformed(o.transform.matrix()));
        }
        Ok(out)
    }
}

impl MeshTools for Blender {
    fn describe(&self) -> String {
        format!("Blender {}", self.version)
    }

    fn apply(&self, mesh: &Mesh, op: MeshOp) -> Result<Mesh, String> {
        match op {
            MeshOp::VoxelRemesh { voxel } => {
                let size = mesh.bounds().map_or(0.0, |b| b.size().max_element());
                if size / voxel > 2000.0 {
                    return Err(format!(
                        "a {:.3} mm voxel is too fine for a {:.0} mm model; use at least {:.3} mm",
                        voxel * 1000.0,
                        size * 1000.0,
                        size * 1000.0 / 2000.0
                    ));
                }
                self.process("voxel_remesh", mesh, &json!({ "voxel": voxel }))
            }
            MeshOp::Decimate { ratio } => {
                self.process("decimate", mesh, &json!({ "ratio": ratio }))
            }
            MeshOp::UvUnwrap { angle_deg, margin } => self.process(
                "uv_unwrap",
                mesh,
                &json!({ "angle_deg": angle_deg, "margin": margin }),
            ),
        }
    }
}

/// A [`Blender`] found on first use, so front ends can start instantly and a
/// missing Blender only matters to the commands that need it. A failed search
/// is retried next time, so installing Blender needs no restart.
#[derive(Debug, Default)]
pub struct LazyBlender {
    found: Mutex<Option<Blender>>,
}

impl LazyBlender {
    pub fn new() -> Self {
        Self::default()
    }

    fn get(&self) -> Result<Blender, String> {
        let mut found = self
            .found
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(b) = found.as_ref() {
            return Ok(b.clone());
        }
        let b = Blender::locate()?;
        tracing::info!(version = b.version(), program = %b.program().display(), "found Blender");
        *found = Some(b.clone());
        Ok(b)
    }
}

impl MeshTools for LazyBlender {
    fn describe(&self) -> String {
        self.found
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map_or_else(|| "Blender".to_owned(), MeshTools::describe)
    }

    fn apply(&self, mesh: &Mesh, op: MeshOp) -> Result<Mesh, String> {
        self.get()?.apply(mesh, op)
    }
}

/// Places to look for Blender, in order.
fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = std::env::var_os("HUB_BLENDER") {
        out.push(PathBuf::from(p));
        return out;
    }
    let exe = if cfg!(windows) {
        "blender.exe"
    } else {
        "blender"
    };
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let p = dir.join(exe);
            if p.is_file() {
                out.push(p);
                break;
            }
        }
    }
    let mut installed: Vec<PathBuf> = Vec::new();
    if cfg!(windows) {
        for root in ["ProgramFiles", "ProgramW6432"] {
            if let Some(base) = std::env::var_os(root) {
                let foundation = PathBuf::from(base).join("Blender Foundation");
                if let Ok(entries) = std::fs::read_dir(&foundation) {
                    installed.extend(entries.flatten().map(|e| e.path().join("blender.exe")));
                }
            }
        }
    } else if cfg!(target_os = "macos") {
        installed.push("/Applications/Blender.app/Contents/MacOS/Blender".into());
    } else {
        installed.extend(["/snap/bin/blender".into(), "/usr/local/bin/blender".into()]);
    }
    // "Blender 5.2" sorts after "Blender 4.5"; newest first.
    installed.sort_by_key(|p| std::cmp::Reverse(parse_version(&p.to_string_lossy())));
    out.extend(installed.into_iter().filter(|p| p.is_file()));
    out.dedup();
    out
}

/// First "major.minor" in `text`, or (0, 0).
fn parse_version(text: &str) -> (u32, u32) {
    let bytes = text.as_bytes();
    for (i, c) in bytes.iter().enumerate() {
        if c.is_ascii_digit() && (i == 0 || !bytes[i - 1].is_ascii_digit()) {
            let rest = &text[i..];
            let mut parts = rest.split(|c: char| !c.is_ascii_digit());
            if let (Some(a), Some(b)) = (parts.next(), parts.next())
                && rest[a.len()..].starts_with('.')
                && let (Ok(a), Ok(b)) = (a.parse(), b.parse())
            {
                return (a, b);
            }
        }
    }
    (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_parse_from_names_and_reports() {
        assert_eq!(parse_version("5.2.3 LTS"), (5, 2));
        assert_eq!(
            parse_version(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"),
            (4, 5)
        );
        assert_eq!(parse_version("blender"), (0, 0));
    }

    #[test]
    fn a_missing_program_is_a_clear_error() {
        let err = Blender::at(Path::new("/nonexistent/blender")).unwrap_err();
        assert!(err.contains("could not start"), "{err}");
    }
}
