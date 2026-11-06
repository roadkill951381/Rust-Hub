//! glTF 2.0 (`.gltf` + `.bin`, or binary `.glb`).

use std::path::Path;

use glam::{Mat4, Vec2, Vec3};
use gltf::mesh::Mode;
use serde_json::{Value, json};

use super::{file_stem, read_file, write_file};
use crate::error::{Error, Result};
use crate::mesh::Mesh;
use crate::scene::{Material, NewObject, Object, Transform};

pub(super) fn import(path: &Path) -> Result<Vec<NewObject>> {
    let err = |message: String| Error::Gltf {
        path: path.to_owned(),
        message,
    };
    let bytes = read_file(path)?;
    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| err(e.to_string()))?;
    let buffers = gltf::import_buffers(&gltf.document, path.parent(), gltf.blob.clone())
        .map_err(|e| err(e.to_string()))?;
    let document = &gltf.document;

    let mut out = Vec::new();
    let roots: Vec<gltf::Node> = match document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        Some(scene) => scene.nodes().collect(),
        // No scene: treat every mesh as placed at the origin.
        None => document.nodes().filter(|n| n.mesh().is_some()).collect(),
    };
    let mut stack: Vec<(gltf::Node, Mat4)> =
        roots.into_iter().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            let primitives: Vec<_> = mesh.primitives().collect();
            let base_name = node
                .name()
                .or(mesh.name())
                .map_or_else(|| file_stem(path), str::to_owned);
            for (i, primitive) in primitives.iter().enumerate() {
                let Some(m) = read_primitive(primitive, &buffers).map_err(err)? else {
                    continue;
                };
                let name = if primitives.len() > 1 {
                    format!("{base_name}.{i}")
                } else {
                    base_name.clone()
                };
                out.push(NewObject {
                    name,
                    mesh: m,
                    transform: Transform::from_matrix(world),
                    material: read_material(&primitive.material()),
                });
            }
        }
        stack.extend(node.children().map(|c| (c, world)));
    }
    // Stack order is reversed; restore document order for stable ids.
    out.reverse();
    Ok(out)
}

fn read_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> std::result::Result<Option<Mesh>, String> {
    let mode = primitive.mode();
    if !matches!(
        mode,
        Mode::Triangles | Mode::TriangleStrip | Mode::TriangleFan
    ) {
        return Ok(None); // points and lines carry no surface
    }
    let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
    let Some(positions) = reader.read_positions() else {
        return Ok(None);
    };
    let positions: Vec<Vec3> = positions.map(Vec3::from_array).collect();
    let normals: Vec<Vec3> = reader
        .read_normals()
        .map(|n| n.map(Vec3::from_array).collect())
        .unwrap_or_default();
    let uvs: Vec<Vec2> = reader
        .read_tex_coords(0)
        .map(|t| t.into_f32().map(Vec2::from_array).collect())
        .unwrap_or_default();
    let raw: Vec<u32> = match reader.read_indices() {
        Some(i) => i.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };
    let indices = match mode {
        Mode::TriangleStrip => (0..raw.len().saturating_sub(2))
            .flat_map(|i| {
                if i % 2 == 0 {
                    [raw[i], raw[i + 1], raw[i + 2]]
                } else {
                    [raw[i + 1], raw[i], raw[i + 2]]
                }
            })
            .collect(),
        Mode::TriangleFan => (1..raw.len().saturating_sub(1))
            .flat_map(|i| [raw[0], raw[i], raw[i + 1]])
            .collect(),
        _ => raw,
    };
    let mesh = Mesh {
        normals: if normals.len() == positions.len() {
            normals
        } else {
            Vec::new()
        },
        uvs: if uvs.len() == positions.len() {
            uvs
        } else {
            Vec::new()
        },
        positions,
        indices,
    };
    mesh.validate()?;
    Ok(Some(mesh))
}

fn read_material(material: &gltf::Material) -> Material {
    let pbr = material.pbr_metallic_roughness();
    Material {
        base_color: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
    }
}

// glTF enum values.
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;

/// Builds the binary buffer and JSON for `objects`, then writes `.glb` or
/// `.gltf` plus a sibling `.bin`.
pub(super) fn export(path: &Path, objects: &[&Object], binary: bool) -> Result<()> {
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut accessors = Vec::new();
    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    let mut nodes = Vec::new();

    // Appends a tightly packed f32/u32 block and returns its accessor index.
    let mut push = |bytes: &[u8],
                    count: usize,
                    component: u32,
                    kind: &str,
                    target: u32,
                    min_max: Option<(Vec<f32>, Vec<f32>)>|
     -> usize {
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let offset = bin.len();
        bin.extend_from_slice(bytes);
        views.push(json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": bytes.len(),
            "target": target,
        }));
        let mut accessor = json!({
            "bufferView": views.len() - 1,
            "componentType": component,
            "count": count,
            "type": kind,
        });
        if let Some((min, max)) = min_max {
            accessor["min"] = json!(min);
            accessor["max"] = json!(max);
        }
        accessors.push(accessor);
        accessors.len() - 1
    };

    for object in objects {
        let mesh = &object.mesh;
        if mesh.positions.is_empty() {
            continue;
        }
        let bounds = mesh.bounds().expect("non-empty mesh has bounds");
        let mut attributes = serde_json::Map::new();
        let position = push(
            &f32_bytes(mesh.positions.iter().flat_map(glam::Vec3::to_array)),
            mesh.positions.len(),
            FLOAT,
            "VEC3",
            ARRAY_BUFFER,
            Some((
                bounds.min.to_array().to_vec(),
                bounds.max.to_array().to_vec(),
            )),
        );
        attributes.insert("POSITION".into(), json!(position));
        if !mesh.normals.is_empty() {
            let normal = push(
                &f32_bytes(mesh.normals.iter().flat_map(glam::Vec3::to_array)),
                mesh.normals.len(),
                FLOAT,
                "VEC3",
                ARRAY_BUFFER,
                None,
            );
            attributes.insert("NORMAL".into(), json!(normal));
        }
        if !mesh.uvs.is_empty() {
            let uv = push(
                &f32_bytes(mesh.uvs.iter().flat_map(glam::Vec2::to_array)),
                mesh.uvs.len(),
                FLOAT,
                "VEC2",
                ARRAY_BUFFER,
                None,
            );
            attributes.insert("TEXCOORD_0".into(), json!(uv));
        }
        let indices = push(
            &mesh
                .indices
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
            mesh.indices.len(),
            UNSIGNED_INT,
            "SCALAR",
            ELEMENT_ARRAY_BUFFER,
            None,
        );

        let m = object.material;
        materials.push(json!({
            "name": object.name,
            "pbrMetallicRoughness": {
                "baseColorFactor": m.base_color,
                "metallicFactor": m.metallic,
                "roughnessFactor": m.roughness,
            },
        }));
        meshes.push(json!({
            "name": object.name,
            "primitives": [{
                "attributes": attributes,
                "indices": indices,
                "material": materials.len() - 1,
                "mode": 4,
            }],
        }));
        let t = object.transform;
        nodes.push(json!({
            "name": object.name,
            "mesh": meshes.len() - 1,
            "translation": t.translation.to_array(),
            "rotation": t.rotation.to_array(),
            "scale": t.scale.to_array(),
        }));
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    let bin_name = format!("{}.bin", file_stem(path));
    let mut buffer = json!({ "byteLength": bin.len() });
    if !binary {
        buffer["uri"] = Value::String(bin_name.clone());
    }
    let root = json!({
        "asset": { "version": "2.0", "generator": concat!("Rust-Hub ", env!("CARGO_PKG_VERSION")) },
        "scene": 0,
        "scenes": [{ "nodes": (0..nodes.len()).collect::<Vec<_>>() }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [buffer],
    });
    let json_bytes = serde_json::to_vec(&root).expect("glTF JSON serializes");

    if binary {
        write_file(path, &glb(json_bytes, &bin))
    } else {
        write_file(&path.with_file_name(&bin_name), &bin)?;
        write_file(path, &json_bytes)
    }
}

fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}

/// Packs a GLB container: header, JSON chunk (space-padded), BIN chunk (zero-padded).
fn glb(mut json: Vec<u8>, bin: &[u8]) -> Vec<u8> {
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(bin);
    out
}
