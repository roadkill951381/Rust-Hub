//! A session owns the scene and its undo history and executes commands.

use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};

use crate::analysis;
use crate::command::{Command, ObjectInfo, Outcome, SizeAxis};
use crate::error::{Error, Result};
use crate::io;
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
        }
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

    pub fn execute(&mut self, command: Command) -> Result<Outcome> {
        self.gesture = None;
        self.run(command)
    }

    /// Runs an edit as part of a continuous gesture, such as dragging a value
    /// in the GUI: every call with the same `gesture` id after the first
    /// replaces the scene without adding another undo step, so one undo
    /// reverts the whole drag.
    pub fn execute_coalescing(&mut self, command: Command, gesture: u64) -> Result<Outcome> {
        if self.gesture == Some(gesture) && command.is_edit() {
            let mut next = self.scene.clone();
            let outcome = apply(&mut next, command)?;
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
            Command::CheckPrintable { objects } => self.check(objects.as_deref()),
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
                let outcome = apply(&mut next, edit)?;
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

    fn check(&self, ids: Option<&[ObjectId]>) -> Result<Outcome> {
        let ids = self.scene.resolve(ids)?;
        let reports: Vec<_> = ids
            .iter()
            .map(|&id| self.scene.get(id).map(analysis::check_printable))
            .collect::<Result<_>>()?;
        let ok = reports.iter().filter(|r| r.printable).count();
        Ok(Outcome {
            summary: format!("{ok} of {} object(s) ready to print", reports.len()),
            reports,
            ..Outcome::default()
        })
    }
}

/// Applies an edit to `scene`. Non-edit commands never reach here.
fn apply(scene: &mut Scene, command: Command) -> Result<Outcome> {
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
        Command::Repair { objects } => {
            let ids = scene.resolve(objects.as_deref())?;
            let mut repairs = Vec::new();
            for &id in &ids {
                let o = scene.get_mut(id)?;
                let (mesh, summary) = analysis::repair(&o.mesh);
                o.mesh = Arc::new(mesh);
                repairs.push(summary);
            }
            let reports = ids
                .iter()
                .map(|&id| scene.get(id).map(analysis::check_printable))
                .collect::<Result<Vec<_>>>()?;
            let fixed: usize = repairs
                .iter()
                .map(|r| r.removed_degenerate + r.removed_duplicates + r.flipped_triangles)
                .sum();
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
        Command::Undo
        | Command::Redo
        | Command::ListObjects
        | Command::CheckPrintable { .. }
        | Command::Export { .. } => unreachable!("handled by Session::execute"),
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
}
