//! The scene: objects with meshes, transforms and materials.
//!
//! Conventions, shared by every crate in the hub:
//! - Units are **metres** and the scene is **Y-up**, matching glTF.
//! - Print formats (STL, 3MF) are millimetres and Z-up; the converters in
//!   [`crate::io`] translate at the file boundary, never inside the scene.

use std::fmt;
use std::sync::Arc;

use glam::{EulerRot, Mat4, Quat, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::mesh::{Aabb, Mesh};

/// Stable identifier for an object. Never reused within a session.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct ObjectId(pub u64);

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Translation, rotation and scale, applied in that order (T · R · S).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    pub fn from_matrix(m: Mat4) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// Rotation as XYZ Euler angles in degrees.
    pub fn euler_degrees(&self) -> Vec3 {
        let (x, y, z) = self.rotation.to_euler(EulerRot::XYZ);
        Vec3::new(x, y, z) * (180.0 / std::f32::consts::PI)
    }

    pub fn rotation_from_degrees(deg: Vec3) -> Quat {
        let r = deg * (std::f32::consts::PI / 180.0);
        Quat::from_euler(EulerRot::XYZ, r.x, r.y, r.z)
    }
}

/// A simple metallic-roughness PBR material, the core of glTF's.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Material {
    /// Linear RGBA base colour, each channel 0 to 1.
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.6,
        }
    }
}

/// One mesh placed in the scene.
///
/// The mesh is shared behind an [`Arc`], so cloning a scene (as the undo
/// history does) copies pointers, not geometry. Edits that change geometry
/// replace the `Arc`, which also tells renderers to re-upload it.
#[derive(Debug, Clone, PartialEq)]
pub struct Object {
    pub id: ObjectId,
    pub name: String,
    pub mesh: Arc<Mesh>,
    pub transform: Transform,
    pub material: Material,
}

impl Object {
    pub fn world_bounds(&self) -> Option<Aabb> {
        self.mesh.bounds_with(self.transform.matrix())
    }

    pub fn world_mesh(&self) -> Mesh {
        self.mesh.transformed(self.transform.matrix())
    }
}

/// An object as read from a file, before it receives an id.
#[derive(Debug, Clone)]
pub struct NewObject {
    pub name: String,
    pub mesh: Mesh,
    pub transform: Transform,
    pub material: Material,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scene {
    objects: Vec<Object>,
    next_id: u64,
}

impl Scene {
    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn get(&self, id: ObjectId) -> Result<&Object> {
        self.objects
            .iter()
            .find(|o| o.id == id)
            .ok_or(Error::UnknownObject(id))
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Result<&mut Object> {
        self.objects
            .iter_mut()
            .find(|o| o.id == id)
            .ok_or(Error::UnknownObject(id))
    }

    pub fn add(&mut self, new: NewObject) -> ObjectId {
        let id = ObjectId(self.next_id);
        self.next_id += 1;
        self.objects.push(Object {
            id,
            name: new.name,
            mesh: Arc::new(new.mesh),
            transform: new.transform,
            material: new.material,
        });
        id
    }

    pub fn remove(&mut self, id: ObjectId) -> Result<Object> {
        let index = self
            .objects
            .iter()
            .position(|o| o.id == id)
            .ok_or(Error::UnknownObject(id))?;
        Ok(self.objects.remove(index))
    }

    /// Resolves an optional id list: `None` or an empty list means every object.
    pub fn resolve(&self, ids: Option<&[ObjectId]>) -> Result<Vec<ObjectId>> {
        match ids {
            Some(ids) if !ids.is_empty() => {
                for &id in ids {
                    self.get(id)?;
                }
                Ok(ids.to_vec())
            }
            _ if self.objects.is_empty() => Err(Error::EmptyScene),
            _ => Ok(self.objects.iter().map(|o| o.id).collect()),
        }
    }

    /// World bounds of the given objects together.
    pub fn bounds_of(&self, ids: &[ObjectId]) -> Result<Option<Aabb>> {
        let mut acc: Option<Aabb> = None;
        for &id in ids {
            if let Some(b) = self.get(id)?.world_bounds() {
                acc = Some(acc.map_or(b, |a| a.union(b)));
            }
        }
        Ok(acc)
    }

    pub fn bounds(&self) -> Option<Aabb> {
        self.objects
            .iter()
            .filter_map(Object::world_bounds)
            .reduce(Aabb::union)
    }

    /// The nearest object a world-space ray hits, tested against triangles.
    pub fn pick(&self, origin: Vec3, dir: Vec3) -> Option<ObjectId> {
        let mut best: Option<(f32, ObjectId)> = None;
        for o in &self.objects {
            let Some(box_hit) = o.world_bounds().and_then(|b| b.ray_hit(origin, dir)) else {
                continue;
            };
            if best.is_some_and(|(t, _)| box_hit > t) {
                continue;
            }
            // Test in object space so the mesh is never copied.
            let inv = o.transform.matrix().inverse();
            let (lo, ld) = (inv.transform_point3(origin), inv.transform_vector3(dir));
            let m = &o.mesh;
            for [a, b, c] in m.triangles() {
                let hit = ray_triangle(
                    lo,
                    ld,
                    m.positions[a as usize],
                    m.positions[b as usize],
                    m.positions[c as usize],
                );
                if let Some(t) = hit {
                    // `t` is in units of `ld`, which maps back to `dir` linearly.
                    if best.is_none_or(|(bt, _)| t < bt) {
                        best = Some((t, o.id));
                    }
                }
            }
        }
        best.map(|(_, id)| id)
    }
}

/// Möller–Trumbore ray/triangle intersection, two-sided.
fn ray_triangle(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let (e1, e2) = (b - a, c - a);
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let s = origin - a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 0.0).then_some(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn pick_finds_the_nearest_object() {
        let mut scene = Scene::default();
        let near = scene.add(NewObject {
            name: "near".into(),
            mesh: primitives::cube(1.0),
            transform: Transform {
                translation: Vec3::new(0.0, 0.0, 2.0),
                ..Transform::IDENTITY
            },
            material: Material::default(),
        });
        let _far = scene.add(NewObject {
            name: "far".into(),
            mesh: primitives::cube(1.0),
            transform: Transform::IDENTITY,
            material: Material::default(),
        });
        assert_eq!(
            scene.pick(Vec3::new(0.0, 0.0, 10.0), Vec3::NEG_Z),
            Some(near)
        );
        assert_eq!(scene.pick(Vec3::new(5.0, 0.0, 10.0), Vec3::NEG_Z), None);
    }
}
