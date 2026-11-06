//! Reading and writing model files.
//!
//! | Format | Units | Up axis | Notes |
//! |---|---|---|---|
//! | glTF / GLB | metres | +Y | the hub's working format, keeps transforms and materials |
//! | STL | millimetres | +Z | triangle soup, transforms baked in |
//! | 3MF | millimetres (or as declared) | +Z | welded vertices, which slicers need for manifold checks |

mod gltf;
mod stl;
mod threemf;

use std::path::Path;

use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::mesh::Mesh;
use crate::scene::{NewObject, Object};

/// The file formats the hub reads and writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Glb,
    Gltf,
    Stl,
    #[serde(rename = "3mf")]
    ThreeMf,
}

impl Format {
    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match ext.as_deref() {
            Some("glb") => Ok(Format::Glb),
            Some("gltf") => Ok(Format::Gltf),
            Some("stl") => Ok(Format::Stl),
            Some("3mf") => Ok(Format::ThreeMf),
            _ => Err(Error::UnsupportedFormat {
                path: path.to_owned(),
            }),
        }
    }

    pub fn is_print_format(self) -> bool {
        matches!(self, Format::Stl | Format::ThreeMf)
    }
}

/// Reads every triangle mesh in a file.
pub fn import(path: &Path) -> Result<Vec<NewObject>> {
    let objects = match Format::from_path(path)? {
        Format::Glb | Format::Gltf => gltf::import(path)?,
        Format::Stl => stl::import(path)?,
        Format::ThreeMf => threemf::import(path)?,
    };
    if objects.iter().all(|o| o.mesh.triangle_count() == 0) {
        return Err(Error::NoGeometry {
            path: path.to_owned(),
        });
    }
    for o in &objects {
        o.mesh
            .validate()
            .map_err(|message| Error::InvalidArgument(format!("{}: {message}", path.display())))?;
    }
    tracing::info!(path = %path.display(), objects = objects.len(), "imported");
    Ok(objects)
}

/// Writes the given objects. Print formats bake transforms and convert to
/// millimetres, Z-up.
pub fn export(path: &Path, objects: &[&Object]) -> Result<()> {
    if objects.is_empty() {
        return Err(Error::EmptyScene);
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|source| Error::Write {
            path: dir.to_owned(),
            source,
        })?;
    }
    match Format::from_path(path)? {
        Format::Glb => gltf::export(path, objects, true),
        Format::Gltf => gltf::export(path, objects, false),
        Format::Stl => stl::export(path, &print_meshes(objects)),
        Format::ThreeMf => threemf::export(path, objects, &print_meshes(objects)),
    }?;
    tracing::info!(path = %path.display(), objects = objects.len(), "exported");
    Ok(())
}

/// Scene space (metres, Y-up) to print space (Z-up), without unit change.
pub fn to_print_space(v: Vec3) -> Vec3 {
    Vec3::new(v.x, -v.z, v.y)
}

/// Print space (Z-up) to scene space (Y-up), without unit change.
pub fn from_print_space(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.z, -v.y)
}

/// World-space meshes converted to millimetres, Z-up.
fn print_meshes(objects: &[&Object]) -> Vec<Mesh> {
    objects
        .iter()
        .map(|o| {
            let mut m = o.world_mesh();
            for p in &mut m.positions {
                *p = to_print_space(*p) * 1000.0;
            }
            for n in &mut m.normals {
                *n = to_print_space(*n);
            }
            m
        })
        .collect()
}

/// Converts a mesh read from a print format (Z-up, `unit_to_metres`) to scene space.
fn from_print_mesh(mut mesh: Mesh, unit_to_metres: f32) -> Mesh {
    for p in &mut mesh.positions {
        *p = from_print_space(*p) * unit_to_metres;
    }
    for n in &mut mesh.normals {
        *n = from_print_space(*n);
    }
    mesh
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| Error::Read {
        path: path.to_owned(),
        source,
    })
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).map_err(|source| Error::Write {
        path: path.to_owned(),
        source,
    })
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Object")
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_space_round_trips() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(from_print_space(to_print_space(v)), v);
        // Scene up (+Y) becomes print up (+Z).
        assert_eq!(to_print_space(Vec3::Y), Vec3::Z);
    }

    #[test]
    fn format_detection_is_case_insensitive() {
        assert_eq!(
            Format::from_path(Path::new("a/B.GLB")).unwrap(),
            Format::Glb
        );
        assert_eq!(
            Format::from_path(Path::new("x.3mf")).unwrap(),
            Format::ThreeMf
        );
        assert!(Format::from_path(Path::new("x.obj")).is_err());
    }
}
