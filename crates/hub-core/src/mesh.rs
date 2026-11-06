//! Triangle meshes and the geometry helpers built on them.

use std::collections::HashMap;

use glam::{Mat3, Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// An indexed triangle mesh in the object's local space, in metres.
///
/// `normals` and `uvs` are either empty or exactly as long as `positions`.
/// Vertices may be split along UV or normal seams, as glTF files usually are;
/// geometry checks weld them by position with [`weld`] instead of relying on
/// shared indices.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Mesh {
    pub positions: Vec<Vec3>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub normals: Vec<Vec3>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Iterates over triangles as index triples.
    pub fn triangles(&self) -> impl ExactSizeIterator<Item = [u32; 3]> + '_ {
        self.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]])
    }

    /// Checks the invariants every other function relies on.
    pub fn validate(&self) -> Result<(), String> {
        if !self.indices.len().is_multiple_of(3) {
            return Err(format!(
                "index count {} is not a multiple of 3",
                self.indices.len()
            ));
        }
        let n = self.positions.len();
        if let Some(bad) = self.indices.iter().find(|&&i| i as usize >= n) {
            return Err(format!("index {bad} is out of range for {n} vertices"));
        }
        if !self.normals.is_empty() && self.normals.len() != n {
            return Err("normal count does not match vertex count".into());
        }
        if !self.uvs.is_empty() && self.uvs.len() != n {
            return Err("uv count does not match vertex count".into());
        }
        if self.positions.iter().any(|p| !p.is_finite()) {
            return Err("mesh contains non-finite vertex positions".into());
        }
        Ok(())
    }

    /// Local-space bounding box, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<Aabb> {
        Aabb::from_points(self.positions.iter().copied())
    }

    /// World-space bounding box under `transform`.
    pub fn bounds_with(&self, transform: Mat4) -> Option<Aabb> {
        Aabb::from_points(
            self.positions
                .iter()
                .map(|&p| transform.transform_point3(p)),
        )
    }

    /// Returns a copy with `transform` baked into positions and normals.
    #[must_use]
    pub fn transformed(&self, transform: Mat4) -> Mesh {
        let normal_matrix = Mat3::from_mat4(transform).inverse().transpose();
        let mut out = Mesh {
            positions: self
                .positions
                .iter()
                .map(|&p| transform.transform_point3(p))
                .collect(),
            normals: self
                .normals
                .iter()
                .map(|&n| (normal_matrix * n).normalize_or_zero())
                .collect(),
            uvs: self.uvs.clone(),
            indices: self.indices.clone(),
        };
        // A mirroring transform turns the surface inside out; swap winding to keep it outward.
        if transform.determinant() < 0.0 {
            out.flip_all();
        }
        out
    }

    /// Reverses the winding of every triangle and negates the normals.
    pub fn flip_all(&mut self) {
        for tri in self.indices.chunks_exact_mut(3) {
            tri.swap(1, 2);
        }
        for n in &mut self.normals {
            *n = -*n;
        }
    }

    /// Recomputes smooth, area-weighted vertex normals from the current indices.
    pub fn compute_normals(&mut self) {
        let mut normals = vec![Vec3::ZERO; self.positions.len()];
        for [a, b, c] in self.triangles() {
            let (pa, pb, pc) = (
                self.positions[a as usize],
                self.positions[b as usize],
                self.positions[c as usize],
            );
            let n = (pb - pa).cross(pc - pa);
            normals[a as usize] += n;
            normals[b as usize] += n;
            normals[c as usize] += n;
        }
        for n in &mut normals {
            *n = n.normalize_or_zero();
        }
        self.normals = normals;
    }

    /// Signed volume enclosed by the surface (positive when outward-facing).
    /// Only meaningful for closed meshes.
    pub fn signed_volume(&self) -> f64 {
        self.triangles()
            .map(|[a, b, c]| {
                let (pa, pb, pc) = (
                    self.positions[a as usize].as_dvec3(),
                    self.positions[b as usize].as_dvec3(),
                    self.positions[c as usize].as_dvec3(),
                );
                pa.dot(pb.cross(pc)) / 6.0
            })
            .sum()
    }

    pub fn surface_area(&self) -> f64 {
        self.triangles()
            .map(|[a, b, c]| {
                let (pa, pb, pc) = (
                    self.positions[a as usize].as_dvec3(),
                    self.positions[b as usize].as_dvec3(),
                    self.positions[c as usize].as_dvec3(),
                );
                (pb - pa).cross(pc - pa).length() * 0.5
            })
            .sum()
    }

    /// Appends `other` to this mesh. Attributes missing on either side are dropped.
    pub fn append(&mut self, other: &Mesh) {
        let base = self.positions.len() as u32;
        let keep_normals = (self.normals.len() == self.positions.len()
            || self.positions.is_empty())
            && other.normals.len() == other.positions.len();
        let keep_uvs = (self.uvs.len() == self.positions.len() || self.positions.is_empty())
            && other.uvs.len() == other.positions.len();
        self.positions.extend_from_slice(&other.positions);
        if keep_normals {
            self.normals.extend_from_slice(&other.normals);
        } else {
            self.normals.clear();
        }
        if keep_uvs {
            self.uvs.extend_from_slice(&other.uvs);
        } else {
            self.uvs.clear();
        }
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }
}

/// An axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Option<Self> {
        let mut it = points.into_iter();
        let first = it.next()?;
        Some(it.fold(
            Aabb {
                min: first,
                max: first,
            },
            |b, p| Aabb {
                min: b.min.min(p),
                max: b.max.max(p),
            },
        ))
    }

    #[must_use]
    pub fn union(self, other: Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Ray/box intersection distance along `dir` (slab method), if any.
    pub fn ray_hit(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        let inv = dir.recip();
        let t1 = (self.min - origin) * inv;
        let t2 = (self.max - origin) * inv;
        let near = t1.min(t2).max_element();
        let far = t1.max(t2).min_element();
        (far >= near.max(0.0)).then_some(near.max(0.0))
    }
}

/// Result of welding a mesh's vertices by position.
#[derive(Debug, Clone)]
pub struct Welded {
    /// Unique positions.
    pub positions: Vec<Vec3>,
    /// For each original vertex, the index of its welded position.
    pub remap: Vec<u32>,
}

/// Merges vertices that sit within `tolerance` of each other.
///
/// Uses a hash grid with cell size `tolerance`, checking neighbouring cells, so
/// points that straddle a cell boundary still merge.
pub fn weld(positions: &[Vec3], tolerance: f32) -> Welded {
    let tolerance = tolerance.max(f32::MIN_POSITIVE);
    let inv = 1.0 / f64::from(tolerance);
    let cell = |p: Vec3| -> [i64; 3] {
        let q = p.as_dvec3() * inv;
        [q.x.floor() as i64, q.y.floor() as i64, q.z.floor() as i64]
    };
    let tol2 = tolerance * tolerance;
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::with_capacity(positions.len());
    let mut unique: Vec<Vec3> = Vec::with_capacity(positions.len());
    let mut remap = Vec::with_capacity(positions.len());

    for &p in positions {
        let c = cell(p);
        let mut found = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(bucket) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                        && let Some(&i) = bucket
                            .iter()
                            .find(|&&i| unique[i as usize].distance_squared(p) <= tol2)
                    {
                        found = Some(i);
                        break 'search;
                    }
                }
            }
        }
        let index = found.unwrap_or_else(|| {
            let i = unique.len() as u32;
            unique.push(p);
            grid.entry(c).or_default().push(i);
            i
        });
        remap.push(index);
    }
    Welded {
        positions: unique,
        remap,
    }
}

/// A weld tolerance scaled to the model: one millionth of its bounding-box
/// diagonal, but never below a micrometre.
pub fn default_weld_tolerance(bounds: Option<Aabb>) -> f32 {
    let diagonal = bounds.map_or(1.0, |b| b.size().length());
    (diagonal * 1e-6).max(1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn cube_volume_and_area() {
        let cube = primitives::cube(2.0);
        assert!((cube.signed_volume() - 8.0).abs() < 1e-6);
        assert!((cube.surface_area() - 24.0).abs() < 1e-6);
    }

    #[test]
    fn mirroring_keeps_volume_positive() {
        let cube = primitives::cube(1.0);
        let mirrored = cube.transformed(Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0)));
        assert!(mirrored.signed_volume() > 0.0);
    }

    #[test]
    fn weld_merges_across_cell_boundaries() {
        let tol = 0.01;
        // Two points 0.002 apart that fall into different grid cells.
        let pts = [Vec3::new(0.0099, 0.0, 0.0), Vec3::new(0.0101, 0.0, 0.0)];
        let w = weld(&pts, tol);
        assert_eq!(w.positions.len(), 1);
        assert_eq!(w.remap, vec![0, 0]);
    }

    #[test]
    fn weld_keeps_distinct_points() {
        let pts = [Vec3::ZERO, Vec3::X, Vec3::ZERO];
        let w = weld(&pts, 1e-4);
        assert_eq!(w.positions.len(), 2);
        assert_eq!(w.remap, vec![0, 1, 0]);
    }

    #[test]
    fn ray_hits_box() {
        let b = Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        };
        let t = b.ray_hit(Vec3::new(0.0, 0.0, -5.0), Vec3::Z).unwrap();
        assert!((t - 4.0).abs() < 1e-6);
        assert!(b.ray_hit(Vec3::new(0.0, 3.0, -5.0), Vec3::Z).is_none());
    }
}
