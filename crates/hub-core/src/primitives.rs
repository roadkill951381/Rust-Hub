//! Closed, outward-facing primitive meshes, centred on the origin, in metres.

use std::f32::consts::{PI, TAU};

use glam::{Vec2, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::mesh::Mesh;

/// The primitive shapes the hub can create.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Primitive {
    Cube,
    Sphere,
    Cylinder,
}

impl Primitive {
    /// Builds the shape so its largest dimension equals `size` metres.
    pub fn build(self, size: f32) -> Mesh {
        match self {
            Primitive::Cube => cube(size),
            Primitive::Sphere => uv_sphere(size * 0.5, 48, 24),
            Primitive::Cylinder => cylinder(size * 0.5, size, 48),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Primitive::Cube => "Cube",
            Primitive::Sphere => "Sphere",
            Primitive::Cylinder => "Cylinder",
        }
    }
}

/// A cube with edge length `size`. Each face has its own vertices so normals
/// and UVs stay sharp.
pub fn cube(size: f32) -> Mesh {
    let h = size * 0.5;
    // (normal, u axis, v axis) for each face; u × v = normal keeps winding outward.
    let faces = [
        (Vec3::X, Vec3::NEG_Z, Vec3::Y),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::X, Vec3::NEG_Z),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
    ];
    let mut mesh = Mesh::default();
    for (n, u, v) in faces {
        let base = mesh.positions.len() as u32;
        for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            mesh.positions.push((n + u * su + v * sv) * h);
            mesh.normals.push(n);
            mesh.uvs
                .push(Vec2::new(f32::midpoint(su, 1.0), (1.0 - sv) * 0.5));
        }
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    mesh
}

/// A UV sphere. The UV seam duplicates one column of vertices; the poles are
/// single vertices so no triangle is degenerate.
pub fn uv_sphere(radius: f32, segments: u32, rings: u32) -> Mesh {
    let segments = segments.max(3);
    let rings = rings.max(2);
    let mut mesh = Mesh::default();

    mesh.positions.push(Vec3::Y * radius);
    mesh.normals.push(Vec3::Y);
    mesh.uvs.push(Vec2::new(0.5, 0.0));
    for r in 1..rings {
        let phi = PI * r as f32 / rings as f32;
        for s in 0..=segments {
            let theta = TAU * s as f32 / segments as f32;
            let n = Vec3::new(phi.sin() * theta.cos(), phi.cos(), -phi.sin() * theta.sin());
            mesh.positions.push(n * radius);
            mesh.normals.push(n);
            mesh.uvs.push(Vec2::new(
                s as f32 / segments as f32,
                r as f32 / rings as f32,
            ));
        }
    }
    let bottom = mesh.positions.len() as u32;
    mesh.positions.push(Vec3::NEG_Y * radius);
    mesh.normals.push(Vec3::NEG_Y);
    mesh.uvs.push(Vec2::new(0.5, 1.0));

    let row = segments + 1;
    let ring_start = |r: u32| 1 + (r - 1) * row;
    for s in 0..segments {
        mesh.indices
            .extend_from_slice(&[0, ring_start(1) + s, ring_start(1) + s + 1]);
    }
    for r in 1..rings - 1 {
        for s in 0..segments {
            let a = ring_start(r) + s;
            let b = ring_start(r + 1) + s;
            mesh.indices
                .extend_from_slice(&[a, b, b + 1, a, b + 1, a + 1]);
        }
    }
    let last = ring_start(rings - 1);
    for s in 0..segments {
        mesh.indices
            .extend_from_slice(&[bottom, last + s + 1, last + s]);
    }
    mesh
}

/// A Y-axis cylinder with flat caps.
pub fn cylinder(radius: f32, height: f32, segments: u32) -> Mesh {
    let segments = segments.max(3);
    let h = height * 0.5;
    let mut mesh = Mesh::default();
    let dir = |s: u32| {
        let t = TAU * s as f32 / segments as f32;
        Vec3::new(t.cos(), 0.0, -t.sin())
    };

    // Side wall.
    for s in 0..=segments {
        let d = dir(s);
        let u = s as f32 / segments as f32;
        mesh.positions.push(d * radius + Vec3::Y * h);
        mesh.positions.push(d * radius - Vec3::Y * h);
        mesh.normals.extend_from_slice(&[d, d]);
        mesh.uvs
            .extend_from_slice(&[Vec2::new(u, 0.0), Vec2::new(u, 1.0)]);
    }
    for s in 0..segments {
        let top = 2 * s;
        let (bot, top_next, bot_next) = (top + 1, top + 2, top + 3);
        mesh.indices
            .extend_from_slice(&[top, bot, bot_next, top, bot_next, top_next]);
    }

    // Caps.
    for (y, normal) in [(h, Vec3::Y), (-h, Vec3::NEG_Y)] {
        let center = mesh.positions.len() as u32;
        mesh.positions.push(Vec3::Y * y);
        mesh.normals.push(normal);
        mesh.uvs.push(Vec2::splat(0.5));
        for s in 0..segments {
            let d = dir(s);
            mesh.positions.push(d * radius + Vec3::Y * y);
            mesh.normals.push(normal);
            mesh.uvs.push(Vec2::new(0.5 + d.x * 0.5, 0.5 - d.z * 0.5));
        }
        for s in 0..segments {
            let a = center + 1 + s;
            let b = center + 1 + (s + 1) % segments;
            if normal.y > 0.0 {
                mesh.indices.extend_from_slice(&[center, a, b]);
            } else {
                mesh.indices.extend_from_slice(&[center, b, a]);
            }
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_enclose_positive_volume() {
        let sphere = uv_sphere(1.0, 64, 32);
        let expected = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(sphere.signed_volume() > 0.97 * expected);
        assert!(sphere.signed_volume() < expected);

        let cyl = cylinder(1.0, 2.0, 64);
        let expected = std::f64::consts::PI * 2.0;
        assert!(cyl.signed_volume() > 0.99 * expected);
        assert!(cyl.signed_volume() < expected);
    }

    #[test]
    fn primitives_are_valid() {
        for p in [Primitive::Cube, Primitive::Sphere, Primitive::Cylinder] {
            let m = p.build(1.0);
            m.validate().unwrap();
            let size = m.bounds().unwrap().size();
            assert!((size.max_element() - 1.0).abs() < 1e-5, "{p:?} size {size}");
        }
    }
}
