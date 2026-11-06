//! STL, binary or ASCII on read, binary on write. Assumed millimetres, Z-up.

use std::io::Cursor;
use std::path::Path;

use glam::Vec3;
use stl_io::{Normal, Triangle, Vertex};

use super::{file_stem, from_print_mesh, read_file, write_file};
use crate::error::{Error, Result};
use crate::mesh::Mesh;
use crate::scene::{Material, NewObject, Transform};

pub(super) fn import(path: &Path) -> Result<Vec<NewObject>> {
    let bytes = read_file(path)?;
    let stl = stl_io::read_stl(&mut Cursor::new(bytes)).map_err(|e| Error::Stl {
        path: path.to_owned(),
        message: e.to_string(),
    })?;
    let mesh = Mesh {
        positions: stl.vertices.iter().map(|v| Vec3::from_array(v.0)).collect(),
        normals: Vec::new(), // shaded flat by the renderer; STL normals are often wrong
        uvs: Vec::new(),
        indices: stl
            .faces
            .iter()
            .flat_map(|f| f.vertices.map(|i| i as u32))
            .collect(),
    };
    Ok(vec![NewObject {
        name: file_stem(path),
        mesh: from_print_mesh(mesh, 0.001),
        transform: Transform::IDENTITY,
        material: Material::default(),
    }])
}

/// Writes all meshes, already in print space, as one binary STL.
pub(super) fn export(path: &Path, meshes: &[Mesh]) -> Result<()> {
    let triangles: Vec<Triangle> = meshes
        .iter()
        .flat_map(|m| {
            m.triangles().map(move |t| {
                let [a, b, c] = t.map(|i| m.positions[i as usize]);
                let n = (b - a).cross(c - a).normalize_or_zero();
                Triangle {
                    normal: Normal::new(n.to_array()),
                    vertices: [a, b, c].map(|p| Vertex::new(p.to_array())),
                }
            })
        })
        .collect();
    let mut out = Vec::new();
    stl_io::write_stl(&mut out, triangles.iter()).map_err(|source| Error::Write {
        path: path.to_owned(),
        source,
    })?;
    write_file(path, &out)
}
