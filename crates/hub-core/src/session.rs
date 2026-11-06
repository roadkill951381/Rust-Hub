//! A session owns the scene and its undo history and executes commands.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};

use crate::analysis::{self, CheckOptions, RepairOptions};
use crate::command::{Command, ObjectInfo, Outcome, SavedAsset, SizeAxis};
use crate::engine::{Engines, GenerateRequest, Generated, MeshOp, Progress, RigReport, RigRequest};
use crate::error::{Error, Result};
use crate::hollow;
use crate::io;
use crate::library::{AssetSource, Library, NewVersion};
use crate::mesh::Mesh;
use crate::scene::{NewObject, Object, ObjectId, Scene, Transform};

const MM: f32 = 0.001;
const DEFAULT_HISTORY: usize = 200;

/// The scene plus undo/redo. All front ends drive the hub through [`Session::execute`].
///
/// Each edit runs on a copy of the scene and is committed only if it succeeds,
/// so a failing command never leaves a half-applied change. Copies are cheap:
/// meshes are shared behind `Arc` until an edit replaces them.
#[derive(Debug, Clone)]
pub struct Session {
    scene: Scene,
    undo: Vec<(&'static str, Scene)>,
    redo: Vec<(&'static str, Scene)>,
    history_limit: usize,
    revision: u64,
    /// Gesture id of the last coalesced edit; see [`Session::execute_coalescing`].
    gesture: Option<u64>,
    engines: Engines,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Self {
            scene: Scene::default(),
            undo: Vec::new(),
            redo: Vec::new(),
            history_limit: DEFAULT_HISTORY,
            revision: 0,
            gesture: None,
            engines: Engines::default(),
        }
    }

    /// A session that can use the given engines (Blender, the generator, the library).
    pub fn with_engines(engines: Engines) -> Self {
        Self {
            engines,
            ..Self::new()
        }
    }

    pub fn engines(&self) -> &Engines {
        &self.engines
    }

    pub fn set_engines(&mut self, engines: Engines) {
        self.engines = engines;
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Increments whenever the scene changes, including undo and redo.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Labels of the edits that undo would revert, most recent last.
    pub fn undo_labels(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.undo.iter().map(|(label, _)| *label)
    }

    /// Runs a command to completion. Long jobs (see [`Command::is_job`]) block;
    /// a GUI should use [`Session::begin`] for those instead.
    pub fn execute(&mut self, command: Command) -> Result<Outcome> {
        self.gesture = None;
        self.run(command)
    }

    /// Starts a command. Quick commands finish immediately; long ones return a
    /// [`Job`] to [`run`](Job::run) on another thread and hand back to
    /// [`Session::finish`], which commits the result as one undo step.
    /// The session stays usable while the job runs.
    pub fn begin(&mut self, command: Command) -> Result<Step> {
        self.gesture = None;
        if command.is_job() {
            self.prepare(command).map(Step::Job)
        } else {
            self.run(command).map(Step::Done)
        }
    }

    /// Commits a finished job.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "a job's output is committed once"
    )]
    pub fn finish(&mut self, output: JobOutput) -> Result<Outcome> {
        let mut next = self.scene.clone();
        let outcome = output.apply(&mut next, &self.engines)?;
        self.commit(output.label, next);
        Ok(outcome)
    }

    /// Runs an edit as part of a continuous gesture, such as dragging a value
    /// in the GUI: every call with the same `gesture` id after the first
    /// replaces the scene without adding another undo step, so one undo
    /// reverts the whole drag.
    pub fn execute_coalescing(&mut self, command: Command, gesture: u64) -> Result<Outcome> {
        if self.gesture == Some(gesture) && command.is_edit() {
            let mut next = self.scene.clone();
            let outcome = apply(&mut next, command, &self.engines)?;
            self.scene = next;
            self.revision += 1;
            return Ok(outcome);
        }
        let outcome = self.run(command)?;
        self.gesture = Some(gesture);
        Ok(outcome)
    }

    fn run(&mut self, command: Command) -> Result<Outcome> {
        tracing::debug!(?command, "execute");
        match command {
            Command::Undo => self.undo(),
            Command::Redo => self.redo(),
            Command::ListObjects => Ok(self.list()),
            Command::CheckPrintable {
                objects,
                min_wall_mm,
            } => self.check(objects.as_deref(), min_wall_mm),
            Command::SaveAsset {
                name,
                objects,
                note,
            } => self.save_asset(&name, objects.as_deref(), note),
            Command::ListAssets { query } => {
                let assets = library(&self.engines)?.list(query.as_deref())?;
                Ok(Outcome {
                    summary: format!("{} asset(s) in the library", assets.len()),
                    assets,
                    ..Outcome::default()
                })
            }
            job if job.is_job() => {
                let job = self.prepare(job)?;
                let output = job.run(&|_, _| {})?;
                self.finish(output)
            }
            Command::Export { path, objects } => {
                let ids = self.scene.resolve(objects.as_deref())?;
                let objs: Vec<&Object> = ids
                    .iter()
                    .map(|&id| self.scene.get(id))
                    .collect::<Result<_>>()?;
                io::export(&path, &objs)?;
                let triangles: usize = objs.iter().map(|o| o.mesh.triangle_count()).sum();
                Ok(Outcome {
                    summary: format!(
                        "Exported {} object(s), {triangles} triangles, to {}",
                        objs.len(),
                        path.display()
                    ),
                    path: Some(path),
                    ..Outcome::default()
                })
            }
            edit => {
                let label = edit.label();
                let mut next = self.scene.clone();
                let outcome = apply(&mut next, edit, &self.engines)?;
                self.commit(label, next);
                Ok(outcome)
            }
        }
    }

    fn commit(&mut self, label: &'static str, next: Scene) {
        let previous = std::mem::replace(&mut self.scene, next);
        self.undo.push((label, previous));
        if self.undo.len() > self.history_limit {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.revision += 1;
    }

    fn undo(&mut self) -> Result<Outcome> {
        let (label, previous) = self.undo.pop().ok_or(Error::NothingToUndo)?;
        let current = std::mem::replace(&mut self.scene, previous);
        self.redo.push((label, current));
        self.revision += 1;
        Ok(Outcome::summary(format!("Undid {label}")))
    }

    fn redo(&mut self) -> Result<Outcome> {
        let (label, next) = self.redo.pop().ok_or(Error::NothingToRedo)?;
        let current = std::mem::replace(&mut self.scene, next);
        self.undo.push((label, current));
        self.revision += 1;
        Ok(Outcome::summary(format!("Redid {label}")))
    }

    fn list(&self) -> Outcome {
        let objects: Vec<ObjectInfo> = self.scene.objects().iter().map(info).collect();
        Outcome {
            summary: format!("{} object(s) in the scene", objects.len()),
            objects,
            ..Outcome::default()
        }
    }

    fn check(&self, ids: Option<&[ObjectId]>, min_wall_mm: Option<f32>) -> Result<Outcome> {
        let ids = self.scene.resolve(ids)?;
        let mut options = CheckOptions::default();
        if let Some(w) = min_wall_mm {
            positive(w, "min_wall_mm")?;
            options.min_wall_mm = w;
        }
        let reports: Vec<_> = ids
            .iter()
            .map(|&id| {
                self.scene
                    .get(id)
                    .map(|o| analysis::check_printable_with(o, &options))
            })
            .collect::<Result<_>>()?;
        let ok = reports.iter().filter(|r| r.printable).count();
        Ok(Outcome {
            summary: format!("{ok} of {} object(s) ready to print", reports.len()),
            reports,
            ..Outcome::default()
        })
    }
}

impl Session {
    fn save_asset(
        &self,
        name: &str,
        ids: Option<&[ObjectId]>,
        note: Option<String>,
    ) -> Result<Outcome> {
        let library = library(&self.engines)?;
        let ids = self.scene.resolve(ids)?;
        let objects: Vec<&Object> = ids
            .iter()
            .map(|&id| self.scene.get(id))
            .collect::<Result<_>>()?;
        let new = NewVersion {
            note,
            source: AssetSource::Saved,
            triangles: objects.iter().map(|o| o.mesh.triangle_count()).sum(),
            size_mm: size_mm(self.scene.bounds_of(&ids)?),
        };
        let (asset, path) = library.save(name, new, |p| io::export(p, &objects))?;
        let version = asset.latest().map_or(1, |v| v.version);
        Ok(Outcome {
            summary: format!(
                "Saved {} version {version} to {}",
                asset.name,
                path.display()
            ),
            saved: Some(SavedAsset {
                name: asset.name,
                version,
                path: path.clone(),
            }),
            path: Some(path),
            ..Outcome::default()
        })
    }

    /// Validates a long command and captures what its job needs.
    fn prepare(&self, command: Command) -> Result<Job> {
        let label = command.label();
        let work = match command {
            Command::Hollow {
                objects,
                wall_mm,
                voxel_mm,
            } => {
                positive(wall_mm, "wall_mm")?;
                if let Some(v) = voxel_mm {
                    positive(v, "voxel_mm")?;
                }
                let op = MeshWork::Hollow {
                    wall: wall_mm * MM,
                    voxel: voxel_mm.map(|v| v * MM),
                };
                Work::Mesh(
                    self.targets(objects.as_deref())?
                        .into_iter()
                        .map(|t| (t, op))
                        .collect(),
                )
            }
            Command::Remesh { objects, voxel_mm } => {
                positive(voxel_mm, "voxel_mm")?;
                mesh_tools(&self.engines)?;
                let op = MeshWork::Tool(MeshOp::VoxelRemesh {
                    voxel: voxel_mm * MM,
                });
                Work::Mesh(
                    self.targets(objects.as_deref())?
                        .into_iter()
                        .map(|t| (t, op))
                        .collect(),
                )
            }
            Command::Decimate {
                objects,
                ratio,
                triangles,
            } => {
                mesh_tools(&self.engines)?;
                let targets = self.targets(objects.as_deref())?;
                let mut items = Vec::with_capacity(targets.len());
                for t in targets {
                    let r = match (ratio, triangles) {
                        (Some(r), None) if r > 0.0 && r <= 1.0 => r,
                        (Some(_), None) => {
                            return Err(Error::InvalidArgument(
                                "ratio must be greater than 0 and at most 1".into(),
                            ));
                        }
                        (None, Some(n)) if n >= 4 => {
                            (n as f32 / t.mesh.triangle_count().max(1) as f32).min(1.0)
                        }
                        (None, Some(_)) => {
                            return Err(Error::InvalidArgument(
                                "triangles must be at least 4".into(),
                            ));
                        }
                        _ => {
                            return Err(Error::InvalidArgument(
                                "give either ratio or triangles".into(),
                            ));
                        }
                    };
                    items.push((t, MeshWork::Tool(MeshOp::Decimate { ratio: r })));
                }
                Work::Mesh(items)
            }
            Command::UvUnwrap {
                objects,
                angle_deg,
                margin,
            } => {
                mesh_tools(&self.engines)?;
                let angle_deg = angle_deg.unwrap_or(66.0);
                let margin = margin.unwrap_or(0.02);
                if !(1.0..=89.0).contains(&angle_deg) || !(0.0..=0.5).contains(&margin) {
                    return Err(Error::InvalidArgument(
                        "angle_deg must be 1 to 89 and margin 0 to 0.5".into(),
                    ));
                }
                let op = MeshWork::Tool(MeshOp::UvUnwrap { angle_deg, margin });
                Work::Mesh(
                    self.targets(objects.as_deref())?
                        .into_iter()
                        .map(|t| (t, op))
                        .collect(),
                )
            }
            Command::Generate {
                prompt,
                image,
                model,
                image_model,
                texture,
                seed,
                faces,
                name,
                size_mm,
                face,
                multiview,
                rig_like,
                materials,
            } => {
                generator(&self.engines)?;
                let prompt = prompt.filter(|p| !p.trim().is_empty());
                if prompt.is_none() && image.is_none() {
                    return Err(Error::InvalidArgument(
                        "give a prompt describing the object, or an image of it".into(),
                    ));
                }
                if face.is_some() && (image.is_some() || prompt.is_none()) {
                    return Err(Error::InvalidArgument(
                        "a face is kept while a picture is drawn from the prompt: give a \
                         prompt and no image"
                            .into(),
                    ));
                }
                for path in [&image, &face, &rig_like].into_iter().flatten() {
                    must_exist(path)?;
                }
                if let Some(s) = size_mm {
                    positive(s, "size_mm")?;
                }
                if rig_like.is_some() {
                    rigger(&self.engines)?;
                }
                Work::Generate {
                    request: GenerateRequest {
                        prompt,
                        image,
                        model,
                        image_model,
                        // A rigged character needs its colours to split skin from clothes.
                        texture: texture || rig_like.is_some(),
                        seed,
                        faces,
                        face,
                        multiview,
                        pose: None,
                    },
                    name,
                    size_mm,
                    rig: rig_like.map(|reference| (reference, materials)),
                }
            }
            Command::Rig {
                model,
                reference,
                output,
                triangles,
                materials,
            } => {
                rigger(&self.engines)?;
                must_exist(&model)?;
                must_exist(&reference)?;
                Work::Rig(RigRequest {
                    model,
                    reference,
                    output,
                    triangles,
                    materials,
                })
            }
            other => unreachable!("{} is not a job", other.label()),
        };
        Ok(Job {
            label,
            work,
            engines: self.engines.clone(),
        })
    }

    fn targets(&self, ids: Option<&[ObjectId]>) -> Result<Vec<Target>> {
        self.scene
            .resolve(ids)?
            .into_iter()
            .map(|id| {
                let o = self.scene.get(id)?;
                Ok(Target {
                    id,
                    name: o.name.clone(),
                    mesh: Arc::clone(&o.mesh),
                    transform: o.transform.matrix(),
                })
            })
            .collect()
    }
}

/// What [`Session::begin`] started.
#[derive(Debug)]
pub enum Step {
    /// A quick command, already applied.
    Done(Outcome),
    /// A long command to run with [`Job::run`] and commit with [`Session::finish`].
    Job(Job),
}

/// A long-running command, detached from the session so it can run on
/// another thread while the scene stays editable.
#[derive(Debug)]
pub struct Job {
    label: &'static str,
    work: Work,
    engines: Engines,
}

#[derive(Debug)]
enum Work {
    Mesh(Vec<(Target, MeshWork)>),
    Generate {
        request: GenerateRequest,
        name: Option<String>,
        size_mm: Option<f32>,
        /// Rig the result like this character, with these material names.
        rig: Option<(PathBuf, Option<String>)>,
    },
    Rig(RigRequest),
}

#[derive(Debug, Clone, Copy)]
enum MeshWork {
    Hollow { wall: f32, voxel: Option<f32> },
    Tool(MeshOp),
}

#[derive(Debug)]
struct Target {
    id: ObjectId,
    name: String,
    mesh: Arc<Mesh>,
    transform: Mat4,
}

/// A finished job, ready for [`Session::finish`].
#[derive(Debug)]
pub struct JobOutput {
    label: &'static str,
    result: WorkResult,
}

#[derive(Debug)]
enum WorkResult {
    Meshes(Vec<Replacement>),
    Generated {
        generated: Generated,
        name: Option<String>,
        size_mm: Option<f32>,
    },
    Rigged {
        request: RigRequest,
        report: RigReport,
    },
}

#[derive(Debug)]
struct Replacement {
    id: ObjectId,
    before: Arc<Mesh>,
    after: Mesh,
    detail: String,
}

impl Job {
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Does the slow part. Touches no session state, so it can run anywhere.
    pub fn run(self, progress: Progress<'_>) -> Result<JobOutput> {
        let result = match self.work {
            Work::Mesh(items) => {
                let count = items.len();
                let mut replaced = Vec::with_capacity(count);
                for (i, (target, op)) in items.into_iter().enumerate() {
                    progress(
                        i as f32 / count as f32,
                        &format!("{}: {}", self.label, target.name),
                    );
                    let world = target.mesh.transformed(target.transform);
                    let (after, detail) = match op {
                        MeshWork::Hollow { wall, voxel } => hollowed(&world, wall, voxel, &target)?,
                        MeshWork::Tool(op) => {
                            let tools = mesh_tools(&self.engines)?;
                            let out = tools.apply(&world, op).map_err(Error::Engine)?;
                            out.validate().map_err(|e| {
                                Error::Engine(format!(
                                    "{} returned a broken mesh: {e}",
                                    tools.describe()
                                ))
                            })?;
                            let detail = format!(
                                "{}: {} to {} triangles",
                                target.name,
                                world.triangle_count(),
                                out.triangle_count()
                            );
                            (out, detail)
                        }
                    };
                    replaced.push(Replacement {
                        id: target.id,
                        before: target.mesh,
                        after: after.transformed(target.transform.inverse()),
                        detail,
                    });
                }
                progress(1.0, "done");
                WorkResult::Meshes(replaced)
            }
            Work::Generate {
                request,
                name,
                size_mm,
                rig,
            } => {
                let share = if rig.is_some() { 0.9 } else { 1.0 };
                let mut request = request;
                if let Some((reference, _)) = &rig
                    && request.image.is_none()
                {
                    progress(0.0, "drawing the pose guide in Blender");
                    request.pose = Some(pose_guide(&self.engines, reference, request.multiview)?);
                }
                let mut generated = generator(&self.engines)?
                    .generate(&request, &|f, s| progress(f * share, s))
                    .map_err(Error::Engine)?;
                if let Some((reference, materials)) = rig {
                    progress(share, "rigging like the reference character");
                    let stem = generated
                        .path
                        .file_stem()
                        .map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
                    let request = RigRequest {
                        model: generated.path.clone(),
                        reference,
                        output: generated.path.with_file_name(format!("{stem}-rigged.glb")),
                        triangles: request.faces,
                        materials,
                    };
                    let report = rigger(&self.engines)?
                        .rig(&request)
                        .map_err(Error::Engine)?;
                    generated.path = request.output;
                    generated.info.rig = Some(report);
                }
                WorkResult::Generated {
                    generated,
                    name,
                    size_mm,
                }
            }
            Work::Rig(request) => {
                progress(0.0, "rigging like the reference character");
                let report = rigger(&self.engines)?
                    .rig(&request)
                    .map_err(Error::Engine)?;
                progress(1.0, "done");
                WorkResult::Rigged { request, report }
            }
        };
        Ok(JobOutput {
            label: self.label,
            result,
        })
    }
}

impl JobOutput {
    pub fn label(&self) -> &'static str {
        self.label
    }

    fn apply(&self, scene: &mut Scene, engines: &Engines) -> Result<Outcome> {
        match &self.result {
            WorkResult::Meshes(replaced) => {
                let mut ids = Vec::with_capacity(replaced.len());
                for r in replaced {
                    let o = scene.get_mut(r.id)?;
                    if !Arc::ptr_eq(&o.mesh, &r.before) {
                        return Err(Error::Engine(format!(
                            "object {} was edited while {} was running; run it again",
                            r.id, self.label
                        )));
                    }
                    o.mesh = Arc::new(r.after.clone());
                    ids.push(r.id);
                }
                let details: Vec<&str> = replaced.iter().map(|r| r.detail.as_str()).collect();
                let reports = if self.label == "Hollow" {
                    ids.iter()
                        .map(|&id| scene.get(id).map(analysis::check_printable))
                        .collect::<Result<_>>()?
                } else {
                    Vec::new()
                };
                Ok(Outcome {
                    summary: format!("{}: {}", self.label, details.join("; ")),
                    objects: infos(scene, &ids)?,
                    reports,
                    ..Outcome::default()
                })
            }
            WorkResult::Generated {
                generated,
                name,
                size_mm,
            } => finish_generation(scene, engines, generated, name.as_deref(), *size_mm),
            WorkResult::Rigged { request, report } => finish_rig(scene, engines, request, report),
        }
    }
}

fn finish_rig(
    scene: &mut Scene,
    engines: &Engines,
    request: &RigRequest,
    report: &RigReport,
) -> Result<Outcome> {
    let new = io::import(&request.output)?;
    let created: Vec<ObjectId> = new.into_iter().map(|o| scene.add(o)).collect();
    apply(
        scene,
        Command::PlaceOnFloor {
            objects: Some(created.clone()),
        },
        engines,
    )?;
    Ok(Outcome {
        summary: format!(
            "Rigged {} like {}: {}; wrote {}",
            file_name(&request.model),
            file_name(&request.reference),
            describe_rig(report),
            request.output.display()
        ),
        objects: infos(scene, &created)?,
        created,
        path: Some(request.output.clone()),
        ..Outcome::default()
    })
}

fn file_name(path: &std::path::Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// Renders the reference character as the pose guide for a new picture, into
/// the temp folder (one file per reference and layout, replaced each time).
fn pose_guide(
    engines: &Engines,
    reference: &std::path::Path,
    sheet: bool,
) -> Result<std::path::PathBuf> {
    let dir = std::env::temp_dir().join("rust-hub").join("pose");
    std::fs::create_dir_all(&dir).map_err(|source| Error::Write {
        path: dir.clone(),
        source,
    })?;
    let stem = reference
        .file_stem()
        .map_or_else(|| "reference".into(), |s| s.to_string_lossy().into_owned());
    let layout = if sheet { "sheet" } else { "front" };
    let output = dir.join(format!("{stem}-{layout}.png"));
    rigger(engines)?
        .pose_views(reference, sheet, &output)
        .map_err(Error::Engine)?;
    Ok(output)
}

/// e.g. "38,748 triangles on 52 bones, materials `surv_mei.body` (31% skin) and
/// `surv_mei.clothes`, arms moved 4° and 5°".
fn describe_rig(report: &RigReport) -> String {
    let mut text = format!(
        "{} triangles on {} bones, materials {} ({:.0}% skin)",
        report.triangles,
        report.bones,
        report.materials.join(" and "),
        report.skin_fraction * 100.0
    );
    if !report.arm_offsets_deg.is_empty() {
        let arms: Vec<String> = report
            .arm_offsets_deg
            .iter()
            .map(|(side, deg)| format!("{side} {:.0}°", deg.abs()))
            .collect();
        let _ = write!(
            text,
            ", arms moved into the rest pose ({})",
            arms.join(", ")
        );
    }
    if report.turned {
        text.push_str(", turned round to face forward");
    }
    if let Some(note) = &report.note {
        let _ = write!(text, " ({note})");
    }
    text
}

fn hollowed(
    world: &Mesh,
    wall: f32,
    voxel: Option<f32>,
    target: &Target,
) -> Result<(Mesh, String)> {
    let t = analysis::topology(world);
    if !t.is_watertight() || t.flipped_edges > 0 || world.signed_volume() <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "{} must be watertight and outward-facing to hollow; run repair first",
            target.name
        )));
    }
    let (mut inner, cavity) = hollow::cavity(world, wall, voxel)
        .map_err(|e| Error::InvalidArgument(format!("{}: {e}", target.name)))?;
    if !world.normals.is_empty() {
        inner.compute_normals();
    }
    if !world.uvs.is_empty() {
        inner.uvs = vec![glam::Vec2::ZERO; inner.positions.len()];
    }
    let mut out = world.clone();
    out.append(&inner);
    let detail = format!(
        "{} hollowed with {:.1} mm walls, {:.1} cm³ removed ({:.2} mm grid)",
        target.name,
        wall * 1000.0,
        cavity.volume * 1e6,
        cavity.voxel * 1000.0
    );
    Ok((out, detail))
}

fn finish_generation(
    scene: &mut Scene,
    engines: &Engines,
    generated: &Generated,
    name: Option<&str>,
    size_mm: Option<f32>,
) -> Result<Outcome> {
    let info = &generated.info;
    let name = name
        .map(str::to_owned)
        .or_else(|| info.prompt.as_deref().map(name_from_prompt))
        .unwrap_or_else(|| "Generated".to_owned());
    let mut new = io::import(&generated.path)?;
    let raw_triangles = new.iter().map(|o| o.mesh.triangle_count()).sum();
    let raw_size = size_mm_of(&new);
    if let [single] = new.as_mut_slice() {
        single.name.clone_from(&name);
    }
    let created: Vec<ObjectId> = new.into_iter().map(|o| scene.add(o)).collect();
    if let Some(size_mm) = size_mm {
        apply(
            scene,
            Command::ScaleToSize {
                objects: Some(created.clone()),
                size_mm,
                axis: SizeAxis::Longest,
            },
            engines,
        )?;
    }
    apply(
        scene,
        Command::PlaceOnFloor {
            objects: Some(created.clone()),
        },
        engines,
    )?;

    let saved = match &engines.library {
        Some(library) => {
            let new = NewVersion {
                note: None,
                source: AssetSource::Generated(Box::new(info.clone())),
                triangles: raw_triangles,
                size_mm: raw_size,
            };
            let (asset, path) = library.save(&name, new, |p| {
                std::fs::copy(&generated.path, p)
                    .map(|_| ())
                    .map_err(|source| Error::Write {
                        path: p.to_owned(),
                        source,
                    })
            })?;
            Some(SavedAsset {
                name: asset.name.clone(),
                version: asset.latest().map_or(1, |v| v.version),
                path,
            })
        }
        None => None,
    };
    let mut summary = format!(
        "Generated {name} with {} in {:.0} s (seed {}), {raw_triangles} triangles",
        info.model, info.seconds, info.seed
    );
    if let Some(rig) = &info.rig {
        let _ = write!(summary, "; rigged: {}", describe_rig(rig));
    }
    if let Some(s) = &saved {
        let _ = write!(summary, "; saved as {} version {}", s.name, s.version);
    }
    Ok(Outcome {
        summary,
        objects: infos(scene, &created)?,
        created,
        path: Some(generated.path.clone()),
        saved,
        generation: Some(info.clone()),
        ..Outcome::default()
    })
}

/// A short title-case name from the first words of a prompt: the subject,
/// up to the first word that starts a description ("a stone owl with big
/// eyes" is "Stone Owl").
fn name_from_prompt(prompt: &str) -> String {
    let words: Vec<String> = prompt
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .take_while(|w| {
            !matches!(
                w.to_lowercase().as_str(),
                "with"
                    | "in"
                    | "on"
                    | "at"
                    | "and"
                    | "holding"
                    | "wearing"
                    | "sitting"
                    | "standing"
                    | "for"
                    | "from"
                    | "that"
                    | "which"
            )
        })
        .filter(|w| !matches!(w.to_lowercase().as_str(), "a" | "an" | "the" | "of"))
        .take(4)
        .map(|w| {
            let mut c = w.chars();
            c.next()
                .map(|f| {
                    f.to_uppercase()
                        .chain(c.flat_map(char::to_lowercase))
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();
    if words.is_empty() {
        "Generated".to_owned()
    } else {
        words.join(" ")
    }
}

fn size_mm(bounds: Option<crate::mesh::Aabb>) -> [f32; 3] {
    bounds.map_or([0.0; 3], |b| {
        let s = b.size() * 1000.0;
        [s.x, s.z, s.y]
    })
}

fn size_mm_of(objects: &[NewObject]) -> [f32; 3] {
    let bounds = objects
        .iter()
        .filter_map(|o| o.mesh.bounds_with(o.transform.matrix()))
        .reduce(crate::mesh::Aabb::union);
    size_mm(bounds)
}

fn library(engines: &Engines) -> Result<&Library> {
    engines.library.as_ref().ok_or_else(|| {
        Error::Library("no asset library is configured; set HUB_LIBRARY or pass --library".into())
    })
}

fn mesh_tools(engines: &Engines) -> Result<&dyn crate::engine::MeshTools> {
    engines.mesh_tools.as_deref().ok_or_else(|| {
        Error::Engine(
            "Blender is needed for this and was not found. Install Blender (4.2 or newer) \
             from blender.org, or set HUB_BLENDER to the blender executable"
                .into(),
        )
    })
}

fn rigger(engines: &Engines) -> Result<&dyn crate::engine::Rigger> {
    engines.rigger.as_deref().ok_or_else(|| {
        Error::Engine(
            "Blender is needed to rig characters and was not found. Install Blender (4.2 or \
             newer) from blender.org, or set HUB_BLENDER to the blender executable"
                .into(),
        )
    })
}

fn must_exist(path: &std::path::Path) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(Error::Read {
            path: path.to_owned(),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        })
    }
}

fn generator(engines: &Engines) -> Result<&dyn crate::engine::Generator> {
    engines.generator.as_deref().ok_or_else(|| {
        Error::Engine(
            "no generation service is configured. Start it with `hub-gen serve` on the GPU \
             machine (see services/hub-gen/README.md) and set HUB_GEN_URL if it is not \
             http://127.0.0.1:8765"
                .into(),
        )
    })
}

/// Applies an edit to `scene`. Queries and jobs never reach here.
fn apply(scene: &mut Scene, command: Command, engines: &Engines) -> Result<Outcome> {
    match command {
        Command::Import { path } => {
            let new = io::import(&path)?;
            let created: Vec<ObjectId> = new.into_iter().map(|o| scene.add(o)).collect();
            let triangles: usize = created
                .iter()
                .map(|&id| scene.get(id).map(|o| o.mesh.triangle_count()))
                .sum::<Result<usize>>()?;
            Ok(Outcome {
                summary: format!(
                    "Imported {} object(s), {triangles} triangles, from {}",
                    created.len(),
                    path.display()
                ),
                objects: infos(scene, &created)?,
                created,
                path: Some(path),
                ..Outcome::default()
            })
        }
        Command::AddPrimitive {
            shape,
            size_mm,
            name,
        } => {
            positive(size_mm, "size_mm")?;
            let id = scene.add(NewObject {
                name: name.unwrap_or_else(|| shape.name().to_owned()),
                mesh: shape.build(size_mm * MM),
                transform: Transform::IDENTITY,
                material: crate::scene::Material::default(),
            });
            Ok(Outcome {
                summary: format!("Added {} ({size_mm} mm) as object {id}", shape.name()),
                created: vec![id],
                objects: infos(scene, &[id])?,
                ..Outcome::default()
            })
        }
        Command::Delete { objects } => {
            if objects.is_empty() {
                return Err(Error::InvalidArgument("no objects given to delete".into()));
            }
            for &id in &objects {
                scene.remove(id)?;
            }
            Ok(Outcome::summary(format!(
                "Deleted {} object(s)",
                objects.len()
            )))
        }
        Command::Rename { object, name } => {
            scene.get_mut(object)?.name.clone_from(&name);
            Ok(Outcome::summary(format!(
                "Renamed object {object} to {name}"
            )))
        }
        Command::SetTransform {
            object,
            position_mm,
            rotation_deg,
            scale,
        } => {
            let o = scene.get_mut(object)?;
            if let Some(p) = position_mm {
                o.transform.translation = Vec3::from_array(p) * MM;
            }
            if let Some(r) = rotation_deg {
                o.transform.rotation = Transform::rotation_from_degrees(Vec3::from_array(r));
            }
            if let Some(s) = scale {
                let s = Vec3::from_array(s);
                if s.cmpeq(Vec3::ZERO).any() || !s.is_finite() {
                    return Err(Error::InvalidArgument(
                        "scale must be finite and non-zero".into(),
                    ));
                }
                o.transform.scale = s;
            }
            Ok(Outcome {
                summary: format!("Updated the transform of object {object}"),
                objects: infos(scene, &[object])?,
                ..Outcome::default()
            })
        }
        Command::Translate { objects, offset_mm } => {
            let ids = scene.resolve(objects.as_deref())?;
            let offset = Vec3::from_array(offset_mm) * MM;
            for &id in &ids {
                scene.get_mut(id)?.transform.translation += offset;
            }
            Ok(Outcome {
                summary: format!("Moved {} object(s) by {offset_mm:?} mm", ids.len()),
                objects: infos(scene, &ids)?,
                ..Outcome::default()
            })
        }
        Command::Rotate { objects, degrees } => {
            let ids = scene.resolve(objects.as_deref())?;
            let center = group_bounds(scene, &ids)?.center();
            let rotation = Transform::rotation_from_degrees(Vec3::from_array(degrees));
            transform_about(scene, &ids, center, Mat4::from_quat(rotation))?;
            Ok(Outcome {
                summary: format!("Rotated {} object(s) by {degrees:?} degrees", ids.len()),
                objects: infos(scene, &ids)?,
                ..Outcome::default()
            })
        }
        Command::ScaleToSize {
            objects,
            size_mm,
            axis,
        } => {
            positive(size_mm, "size_mm")?;
            let ids = scene.resolve(objects.as_deref())?;
            let bounds = group_bounds(scene, &ids)?;
            let size = bounds.size();
            let current = match axis {
                SizeAxis::Longest => size.max_element(),
                SizeAxis::X => size.x,
                SizeAxis::Height => size.y,
                SizeAxis::Depth => size.z,
            };
            if current <= f32::EPSILON {
                return Err(Error::InvalidArgument(format!(
                    "the selection has no extent along {axis:?}"
                )));
            }
            let factor = size_mm * MM / current;
            transform_about(
                scene,
                &ids,
                bounds.center(),
                Mat4::from_scale(Vec3::splat(factor)),
            )?;
            Ok(Outcome {
                summary: format!(
                    "Scaled {} object(s) by {factor:.4} so the {} side is {size_mm} mm",
                    ids.len(),
                    format!("{axis:?}").to_lowercase()
                ),
                objects: infos(scene, &ids)?,
                ..Outcome::default()
            })
        }
        Command::PlaceOnFloor { objects } => {
            let ids = scene.resolve(objects.as_deref())?;
            let bounds = group_bounds(scene, &ids)?;
            let center = bounds.center();
            let offset = Vec3::new(-center.x, -bounds.min.y, -center.z);
            for &id in &ids {
                scene.get_mut(id)?.transform.translation += offset;
            }
            Ok(Outcome {
                summary: format!("Placed {} object(s) on the floor at the origin", ids.len()),
                objects: infos(scene, &ids)?,
                ..Outcome::default()
            })
        }
        Command::ApplyTransform { object } => {
            let o = scene.get_mut(object)?;
            o.mesh = Arc::new(o.mesh.transformed(o.transform.matrix()));
            o.transform = Transform::IDENTITY;
            Ok(Outcome::summary(format!(
                "Applied the transform of object {object}"
            )))
        }
        Command::Repair {
            objects,
            fill_holes,
            remove_small_shells,
        } => {
            let ids = scene.resolve(objects.as_deref())?;
            let options = RepairOptions {
                fill_holes: fill_holes.unwrap_or(true),
                remove_small_shells,
            };
            if let Some(f) = remove_small_shells
                && !(0.0..1.0).contains(&f)
            {
                return Err(Error::InvalidArgument(
                    "remove_small_shells must be between 0 and 1".into(),
                ));
            }
            let mut repairs = Vec::new();
            for &id in &ids {
                let o = scene.get_mut(id)?;
                let (mesh, summary) = analysis::repair_with(&o.mesh, &options);
                o.mesh = Arc::new(mesh);
                repairs.push(summary);
            }
            let reports = ids
                .iter()
                .map(|&id| scene.get(id).map(analysis::check_printable))
                .collect::<Result<Vec<_>>>()?;
            let fixed: usize = repairs.iter().map(analysis::RepairSummary::fixes).sum();
            Ok(Outcome {
                summary: format!("Repaired {} object(s), {fixed} triangle fixes", ids.len()),
                repairs,
                reports,
                ..Outcome::default()
            })
        }
        Command::SetMaterial { object, material } => {
            scene.get_mut(object)?.material = material;
            Ok(Outcome::summary(format!(
                "Updated the material of object {object}"
            )))
        }
        Command::LoadAsset { name, version } => {
            let (asset, chosen, path) = library(engines)?.find(&name, version)?;
            let mut new = io::import(&path)?;
            if let [single] = new.as_mut_slice() {
                single.name.clone_from(&asset.name);
            }
            let created: Vec<ObjectId> = new.into_iter().map(|o| scene.add(o)).collect();
            Ok(Outcome {
                summary: format!(
                    "Loaded {} version {} ({} object(s))",
                    asset.name,
                    chosen.version,
                    created.len()
                ),
                objects: infos(scene, &created)?,
                created,
                path: Some(path),
                ..Outcome::default()
            })
        }
        Command::Undo
        | Command::Redo
        | Command::ListObjects
        | Command::CheckPrintable { .. }
        | Command::SaveAsset { .. }
        | Command::ListAssets { .. }
        | Command::Export { .. }
        | Command::Hollow { .. }
        | Command::Remesh { .. }
        | Command::Decimate { .. }
        | Command::UvUnwrap { .. }
        | Command::Generate { .. }
        | Command::Rig { .. } => unreachable!("handled by Session::run"),
    }
}

/// Applies `m` to the given objects about `center`, in world space.
fn transform_about(scene: &mut Scene, ids: &[ObjectId], center: Vec3, m: Mat4) -> Result<()> {
    let around = Mat4::from_translation(center) * m * Mat4::from_translation(-center);
    for &id in ids {
        let o = scene.get_mut(id)?;
        // Callers pass rotations and uniform scales, which keep T·R·S decomposable.
        let (s, r, t) = (around * o.transform.matrix()).to_scale_rotation_translation();
        o.transform = Transform {
            translation: t,
            rotation: Quat::normalize(r),
            scale: s,
        };
    }
    Ok(())
}

fn group_bounds(scene: &Scene, ids: &[ObjectId]) -> Result<crate::mesh::Aabb> {
    scene
        .bounds_of(ids)?
        .ok_or_else(|| Error::InvalidArgument("the selected objects have no geometry".into()))
}

fn positive(value: f32, name: &str) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(format!(
            "{name} must be a positive number"
        )))
    }
}

/// Builds an [`ObjectInfo`] for one object.
pub fn info(o: &Object) -> ObjectInfo {
    let bounds = o.world_bounds();
    let mm = |v: Vec3| (v * 1000.0).to_array();
    ObjectInfo {
        id: o.id,
        name: o.name.clone(),
        triangles: o.mesh.triangle_count(),
        vertices: o.mesh.vertex_count(),
        size_mm: bounds.map_or([0.0; 3], |b| {
            let s = b.size();
            mm(Vec3::new(s.x, s.z, s.y))
        }),
        center_mm: bounds.map_or([0.0; 3], |b| mm(b.center())),
        position_mm: mm(o.transform.translation),
        rotation_deg: o.transform.euler_degrees().to_array(),
        scale: o.transform.scale.to_array(),
    }
}

fn infos(scene: &Scene, ids: &[ObjectId]) -> Result<Vec<ObjectInfo>> {
    ids.iter().map(|&id| scene.get(id).map(info)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::Primitive;
    use crate::scene::Scene;

    fn add(session: &mut Session, shape: Primitive, size_mm: f32) -> ObjectId {
        session
            .execute(Command::AddPrimitive {
                shape,
                size_mm,
                name: None,
            })
            .unwrap()
            .created[0]
    }

    #[test]
    fn undo_and_redo_restore_scenes() {
        let mut s = Session::new();
        let id = add(&mut s, Primitive::Cube, 10.0);
        s.execute(Command::Translate {
            objects: None,
            offset_mm: [5.0, 0.0, 0.0],
        })
        .unwrap();
        assert!((s.scene().get(id).unwrap().transform.translation.x - 0.005).abs() < 1e-7);
        s.execute(Command::Undo).unwrap();
        assert_eq!(s.scene().get(id).unwrap().transform.translation, Vec3::ZERO);
        s.execute(Command::Undo).unwrap();
        assert!(s.scene().is_empty());
        assert!(matches!(
            s.execute(Command::Undo),
            Err(Error::NothingToUndo)
        ));
        s.execute(Command::Redo).unwrap();
        s.execute(Command::Redo).unwrap();
        assert!((s.scene().get(id).unwrap().transform.translation.x - 0.005).abs() < 1e-7);
    }

    #[test]
    fn a_coalesced_gesture_is_one_undo_step() {
        let mut s = Session::new();
        let id = add(&mut s, Primitive::Cube, 10.0);
        for x in 1..=5 {
            s.execute_coalescing(
                Command::SetTransform {
                    object: id,
                    position_mm: Some([x as f32, 0.0, 0.0]),
                    rotation_deg: None,
                    scale: None,
                },
                7,
            )
            .unwrap();
        }
        assert!((s.scene().get(id).unwrap().transform.translation.x - 0.005).abs() < 1e-7);
        s.execute(Command::Undo).unwrap();
        assert_eq!(s.scene().get(id).unwrap().transform.translation, Vec3::ZERO);
        assert_eq!(s.undo_labels().count(), 1); // only the add remains
    }

    #[test]
    fn failed_edits_change_nothing() {
        let mut s = Session::new();
        let id = add(&mut s, Primitive::Cube, 10.0);
        let before = s.scene().clone();
        let revision = s.revision();
        let err = s.execute(Command::Delete {
            objects: vec![id, ObjectId(99)],
        });
        assert!(err.is_err());
        assert_eq!(s.scene(), &before);
        assert_eq!(s.revision(), revision);
    }

    #[test]
    fn scale_to_size_hits_target_on_each_axis() {
        let mut s = Session::new();
        let id = add(&mut s, Primitive::Cylinder, 10.0);
        s.execute(Command::SetTransform {
            object: id,
            position_mm: None,
            rotation_deg: None,
            scale: Some([1.0, 3.0, 1.0]),
        })
        .unwrap();
        for (axis, index) in [
            (SizeAxis::Height, 2),
            (SizeAxis::X, 0),
            (SizeAxis::Longest, 2),
        ] {
            let out = s
                .execute(Command::ScaleToSize {
                    objects: None,
                    size_mm: 80.0,
                    axis,
                })
                .unwrap();
            let size = out.objects[0].size_mm[index];
            assert!((size - 80.0).abs() < 1e-3, "{axis:?}: {size}");
        }
    }

    #[test]
    fn place_on_floor_rests_on_zero() {
        let mut s = Session::new();
        let id = add(&mut s, Primitive::Sphere, 20.0);
        s.execute(Command::Translate {
            objects: None,
            offset_mm: [3.0, -50.0, 7.0],
        })
        .unwrap();
        s.execute(Command::PlaceOnFloor { objects: None }).unwrap();
        let b = s.scene().get(id).unwrap().world_bounds().unwrap();
        assert!(b.min.y.abs() < 1e-7);
        assert!(b.center().x.abs() < 1e-6 && b.center().z.abs() < 1e-6);
    }

    #[test]
    fn rotate_turns_height_into_width() {
        let mut s = Session::new();
        let id = add(&mut s, Primitive::Cylinder, 10.0);
        s.execute(Command::SetTransform {
            object: id,
            position_mm: None,
            rotation_deg: None,
            scale: Some([1.0, 4.0, 1.0]),
        })
        .unwrap();
        let out = s
            .execute(Command::Rotate {
                objects: None,
                degrees: [0.0, 0.0, 90.0],
            })
            .unwrap();
        let [w, _, h] = out.objects[0].size_mm;
        assert!(w > 39.0 && h < 11.0, "w {w} h {h}");
    }

    #[test]
    fn apply_transform_bakes_geometry() {
        let mut s = Session::new();
        let id = add(&mut s, Primitive::Cube, 10.0);
        s.execute(Command::Translate {
            objects: None,
            offset_mm: [10.0, 0.0, 0.0],
        })
        .unwrap();
        let before = s.scene().get(id).unwrap().world_bounds().unwrap();
        s.execute(Command::ApplyTransform { object: id }).unwrap();
        let o = s.scene().get(id).unwrap();
        assert_eq!(o.transform, Transform::IDENTITY);
        let after = o.world_bounds().unwrap();
        assert!(before.min.abs_diff_eq(after.min, 1e-7));
    }

    #[test]
    fn hollow_leaves_a_printable_shell() {
        let mut s = Session::new();
        let id = add(&mut s, Primitive::Cube, 20.0);
        let out = s
            .execute(Command::Hollow {
                objects: None,
                wall_mm: 2.0,
                voxel_mm: Some(0.5),
            })
            .unwrap();
        let report = &out.reports[0];
        assert!(report.printable, "{:?}", report.issues);
        assert_eq!(report.topology.cavities, 1);
        // 20³ minus about 16³ mm³.
        assert!(
            (report.volume_mm3 - 3904.0).abs() < 300.0,
            "{}",
            report.volume_mm3
        );
        assert!(s.scene().get(id).unwrap().mesh.triangle_count() > 12);
        s.execute(Command::Undo).unwrap();
        assert_eq!(s.scene().get(id).unwrap().mesh.triangle_count(), 12);
    }

    #[test]
    fn jobs_run_detached_and_commit_as_one_step() {
        let mut s = Session::new();
        let id = add(&mut s, Primitive::Sphere, 20.0);
        let Step::Job(job) = s
            .begin(Command::Hollow {
                objects: None,
                wall_mm: 2.0,
                voxel_mm: None,
            })
            .unwrap()
        else {
            panic!("hollow is a job");
        };
        // The scene stays editable while the job runs elsewhere.
        let handle = std::thread::spawn(move || job.run(&|_, _| {}));
        s.execute(Command::Translate {
            objects: None,
            offset_mm: [1.0, 0.0, 0.0],
        })
        .unwrap();
        let output = handle.join().unwrap().unwrap();
        s.finish(output).unwrap();
        assert_eq!(s.undo_labels().last(), Some("Hollow"));
        assert!(s.scene().get(id).unwrap().transform.translation.x > 0.0);
    }

    #[test]
    fn jobs_refuse_to_overwrite_newer_edits() {
        let mut s = Session::new();
        add(&mut s, Primitive::Cube, 20.0);
        let Ok(Step::Job(job)) = s.begin(Command::Hollow {
            objects: None,
            wall_mm: 2.0,
            voxel_mm: Some(1.0),
        }) else {
            panic!("hollow is a job");
        };
        let output = job.run(&|_, _| {}).unwrap();
        s.execute(Command::Repair {
            objects: None,
            fill_holes: None,
            remove_small_shells: None,
        })
        .unwrap();
        let err = s.finish(output).unwrap_err().to_string();
        assert!(err.contains("was edited"), "{err}");
    }

    #[test]
    fn engine_commands_explain_missing_engines() {
        let mut s = Session::new();
        add(&mut s, Primitive::Cube, 20.0);
        let err = s
            .execute(Command::Remesh {
                objects: None,
                voxel_mm: 1.0,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("Blender"), "{err}");
        let err = s
            .execute(Command::Generate {
                prompt: Some("a cup".into()),
                image: None,
                model: None,
                image_model: None,
                texture: false,
                seed: None,
                faces: None,
                name: None,
                size_mm: None,
                face: None,
                multiview: false,
                rig_like: None,
                materials: None,
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("generation service"), "{err}");
        let err = s
            .execute(Command::ListAssets { query: None })
            .unwrap_err()
            .to_string();
        assert!(err.contains("library"), "{err}");
    }

    /// Halves triangle counts by dropping every other triangle.
    #[derive(Debug)]
    struct FakeTools;

    impl crate::engine::MeshTools for FakeTools {
        fn describe(&self) -> String {
            "fake".into()
        }

        fn apply(&self, mesh: &Mesh, op: MeshOp) -> std::result::Result<Mesh, String> {
            let MeshOp::Decimate { ratio } = op else {
                return Err("unsupported".into());
            };
            let keep = ((mesh.triangle_count() as f32) * ratio) as usize;
            let mut out = mesh.clone();
            out.indices.truncate(keep * 3);
            Ok(out)
        }
    }

    /// "Generates" a 1 m sphere by writing a GLB.
    #[derive(Debug)]
    struct FakeGenerator(std::path::PathBuf);

    impl crate::engine::Generator for FakeGenerator {
        fn describe(&self) -> String {
            "fake".into()
        }

        fn generate(
            &self,
            request: &GenerateRequest,
            progress: Progress<'_>,
        ) -> std::result::Result<Generated, String> {
            progress(0.5, "shape");
            let mut tmp = Scene::default();
            let id = tmp.add(NewObject {
                name: "mesh".into(),
                mesh: crate::primitives::uv_sphere(0.5, 16, 8),
                transform: Transform::IDENTITY,
                material: crate::scene::Material::default(),
            });
            let path = self.0.join("out.glb");
            io::export(&path, &[tmp.get(id).unwrap()]).map_err(|e| e.to_string())?;
            Ok(Generated {
                path,
                info: crate::engine::GenerationInfo {
                    model: "fake".into(),
                    image_model: None,
                    prompt: request.prompt.clone(),
                    seed: request.seed.unwrap_or(7),
                    textured: request.texture,
                    seconds: 1.0,
                    // Stands in for the picture the shape was made from.
                    image: request.pose.clone(),
                    face: request.face.clone(),
                    ..crate::engine::GenerationInfo::default()
                },
            })
        }
    }

    #[test]
    fn generated_models_are_named_after_their_subject() {
        assert_eq!(
            name_from_prompt("a stone owl statue with big eyes"),
            "Stone Owl Statue"
        );
        assert_eq!(name_from_prompt("The knight, standing"), "Knight");
        assert_eq!(name_from_prompt("with a hat"), "Generated");
    }

    #[test]
    fn generate_adds_saves_and_reloads_models() {
        let dir = tempfile::tempdir().unwrap();
        let engines = Engines {
            mesh_tools: Some(Arc::new(FakeTools)),
            generator: Some(Arc::new(FakeGenerator(dir.path().to_owned()))),
            rigger: None,
            library: Some(Library::open(dir.path().join("library"))),
        };
        let mut s = Session::with_engines(engines);
        let out = s
            .execute(Command::Generate {
                prompt: Some("a small round pebble".into()),
                image: None,
                model: None,
                image_model: None,
                texture: false,
                seed: Some(3),
                faces: None,
                name: None,
                size_mm: Some(40.0),
                face: None,
                multiview: false,
                rig_like: None,
                materials: None,
            })
            .unwrap();
        let id = out.created[0];
        let o = s.scene().get(id).unwrap();
        assert_eq!(o.name, "Small Round Pebble");
        let b = o.world_bounds().unwrap();
        assert!((b.size().max_element() - 0.04).abs() < 1e-5);
        assert!(b.min.y.abs() < 1e-6, "rests on the floor");
        let saved = out.saved.unwrap();
        assert_eq!(
            (saved.name.as_str(), saved.version),
            ("Small Round Pebble", 1)
        );
        assert_eq!(out.generation.unwrap().seed, 3);

        let out = s
            .execute(Command::Decimate {
                objects: None,
                ratio: Some(0.5),
                triangles: None,
            })
            .unwrap();
        assert!(out.summary.contains("to 112 triangles"), "{}", out.summary);

        s.execute(Command::SaveAsset {
            name: "small round pebble".into(),
            objects: None,
            note: Some("decimated".into()),
        })
        .unwrap();
        let listed = s
            .execute(Command::ListAssets {
                query: Some("pebble".into()),
            })
            .unwrap();
        assert_eq!(listed.assets[0].versions.len(), 2);
        let loaded = s
            .execute(Command::LoadAsset {
                name: "Small Round Pebble".into(),
                version: Some(1),
            })
            .unwrap();
        assert_eq!(loaded.objects[0].triangles, 224);
    }

    /// "Rigs" by copying the model to the output.
    #[derive(Debug)]
    struct FakeRigger;

    impl crate::engine::Rigger for FakeRigger {
        fn describe(&self) -> String {
            "fake".into()
        }

        fn pose_views(
            &self,
            _reference: &std::path::Path,
            sheet: bool,
            output: &std::path::Path,
        ) -> std::result::Result<(), String> {
            std::fs::write(output, if sheet { "sheet" } else { "front" }).map_err(|e| e.to_string())
        }

        fn rig(
            &self,
            request: &crate::engine::RigRequest,
        ) -> std::result::Result<crate::engine::RigReport, String> {
            std::fs::copy(&request.model, &request.output).map_err(|e| e.to_string())?;
            Ok(crate::engine::RigReport {
                triangles: 224,
                bones: 52,
                materials: vec![format!(
                    "{}.body",
                    request.materials.as_deref().unwrap_or("ref")
                )],
                turned: true,
                ..crate::engine::RigReport::default()
            })
        }
    }

    fn character(face: Option<std::path::PathBuf>, image: Option<std::path::PathBuf>) -> Command {
        Command::Generate {
            prompt: Some("Mei, a survivor in a red jacket".into()),
            image,
            model: None,
            image_model: None,
            texture: false,
            seed: Some(3),
            faces: Some(20_000),
            name: Some("Mei".into()),
            size_mm: None,
            face,
            multiview: true,
            rig_like: None,
            materials: Some("surv_mei".into()),
        }
    }

    #[test]
    fn generate_can_rig_its_result_like_a_character() {
        let dir = tempfile::tempdir().unwrap();
        let reference = dir.path().join("reference.glb");
        std::fs::write(&reference, b"glTF").unwrap();
        let face = dir.path().join("portrait.png");
        std::fs::write(&face, b"png").unwrap();
        let mut s = Session::with_engines(Engines {
            generator: Some(Arc::new(FakeGenerator(dir.path().to_owned()))),
            ..Engines::default()
        });
        let mut command = character(Some(face.clone()), None);
        if let Command::Generate { rig_like, .. } = &mut command {
            *rig_like = Some(reference.clone());
        }
        let err = s.execute(command.clone()).unwrap_err().to_string();
        assert!(err.to_lowercase().contains("blender"), "{err}");

        s = Session::with_engines(Engines {
            generator: Some(Arc::new(FakeGenerator(dir.path().to_owned()))),
            rigger: Some(Arc::new(FakeRigger)),
            ..Engines::default()
        });
        let out = s.execute(command).unwrap();
        assert!(out.summary.contains("rigged"), "{}", out.summary);
        let info = out.generation.unwrap();
        assert_eq!(info.face.as_deref(), Some(face.as_path()));
        assert!(info.textured, "rigged characters are always textured");
        let pose = info.image.as_deref().expect("a pose guide was drawn");
        assert_eq!(std::fs::read_to_string(pose).unwrap(), "sheet");
        let rig = info.rig.unwrap();
        assert_eq!(rig.materials, ["surv_mei.body"]);
        assert!(dir.path().join("out-rigged.glb").exists());
        assert_eq!(s.scene().objects().len(), 1);

        let err = s
            .execute(character(Some(face), Some(dir.path().join("out.glb"))))
            .unwrap_err()
            .to_string();
        assert!(err.contains("a face is kept"), "{err}");
    }
}
