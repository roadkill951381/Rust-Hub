//! 3MF (core specification): a zip package holding an XML model.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::{Cursor, Read, Write};
use std::path::Path;

use glam::{Mat4, Vec3, Vec4};
use quick_xml::events::{BytesStart, Event};

use super::{file_stem, from_print_mesh, read_file, write_file};
use crate::error::{Error, Result};
use crate::mesh::{Mesh, default_weld_tolerance, weld};
use crate::scene::{Material, NewObject, Object, Transform};

const MODEL_PATH: &str = "3D/3dmodel.model";
const CORE_NS: &str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";

const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>
"#;

const RELS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>
"#;

/// Writes one 3MF object per scene object. `meshes` are already in print space.
///
/// Vertices are welded by position: slicers check manifoldness through shared
/// vertex indices, so split seams would otherwise read as holes.
pub(super) fn export(path: &Path, objects: &[&Object], meshes: &[Mesh]) -> Result<()> {
    let mut xml = String::with_capacity(1 << 16);
    let _ = write!(
        xml,
        r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="{CORE_NS}">
<metadata name="Application">Rust-Hub {}</metadata>
<resources>
"#,
        env!("CARGO_PKG_VERSION")
    );
    let mut ids = Vec::new();
    for (i, (object, mesh)) in objects.iter().zip(meshes).enumerate() {
        if mesh.triangle_count() == 0 {
            continue;
        }
        let id = i + 1;
        ids.push(id);
        let welded = weld(&mesh.positions, default_weld_tolerance(mesh.bounds()));
        let _ = writeln!(
            xml,
            r#"<object id="{id}" type="model" name="{}"><mesh><vertices>"#,
            escape(&object.name)
        );
        for p in &welded.positions {
            let _ = writeln!(xml, r#"<vertex x="{}" y="{}" z="{}"/>"#, p.x, p.y, p.z);
        }
        xml.push_str("</vertices><triangles>\n");
        for t in mesh.triangles() {
            let [a, b, c] = t.map(|i| welded.remap[i as usize]);
            if a == b || b == c || a == c {
                continue; // collapsed by welding; 3MF forbids repeated vertices
            }
            let _ = writeln!(xml, r#"<triangle v1="{a}" v2="{b}" v3="{c}"/>"#);
        }
        xml.push_str("</triangles></mesh></object>\n");
    }
    xml.push_str("</resources>\n<build>\n");
    for id in ids {
        let _ = writeln!(xml, r#"<item objectid="{id}"/>"#);
    }
    xml.push_str("</build>\n</model>\n");

    let io_err = |e: std::io::Error| Error::Write {
        path: path.to_owned(),
        source: e,
    };
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    for (name, body) in [
        ("[Content_Types].xml", CONTENT_TYPES),
        ("_rels/.rels", RELS),
        (MODEL_PATH, xml.as_str()),
    ] {
        zip.start_file(name, options)
            .map_err(|e| io_err(std::io::Error::other(e)))?;
        zip.write_all(body.as_bytes()).map_err(io_err)?;
    }
    let bytes = zip
        .finish()
        .map_err(|e| io_err(std::io::Error::other(e)))?
        .into_inner();
    write_file(path, &bytes)
}

pub(super) fn import(path: &Path) -> Result<Vec<NewObject>> {
    let err = |message: String| Error::ThreeMf {
        path: path.to_owned(),
        message,
    };
    let bytes = read_file(path)?;
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| err(e.to_string()))?;
    let names: Vec<String> = archive
        .file_names()
        .filter_map(|n| n.ok().map(std::borrow::Cow::into_owned))
        .collect();
    let model_name = names
        .iter()
        .find(|n| n.eq_ignore_ascii_case(MODEL_PATH))
        .or_else(|| {
            names
                .iter()
                .find(|n| n.to_ascii_lowercase().ends_with(".model"))
        })
        .cloned()
        .ok_or_else(|| err("no 3D model part in the package".into()))?;
    let mut xml = String::new();
    archive
        .by_name(&model_name)
        .map_err(|e| err(e.to_string()))?
        .read_to_string(&mut xml)
        .map_err(|e| err(e.to_string()))?;
    let model = parse_model(&xml).map_err(err)?;

    let mut out = Vec::new();
    for item in &model.build {
        let mut mesh = Mesh::default();
        model
            .flatten(item.object, item.transform, &mut mesh, 0)
            .map_err(err)?;
        if mesh.triangle_count() == 0 {
            continue;
        }
        let name = model
            .objects
            .get(&item.object)
            .and_then(|o| o.name.clone())
            .unwrap_or_else(|| file_stem(path));
        out.push(NewObject {
            name,
            mesh: from_print_mesh(mesh, model.unit_to_metres),
            transform: Transform::IDENTITY,
            material: Material::default(),
        });
    }
    Ok(out)
}

#[derive(Default)]
struct Model {
    unit_to_metres: f32,
    objects: HashMap<u32, ModelObject>,
    build: Vec<Item>,
}

#[derive(Default)]
struct ModelObject {
    name: Option<String>,
    mesh: Mesh,
    components: Vec<Item>,
}

struct Item {
    object: u32,
    transform: Mat4,
}

impl Model {
    fn flatten(
        &self,
        id: u32,
        transform: Mat4,
        out: &mut Mesh,
        depth: u32,
    ) -> std::result::Result<(), String> {
        if depth > 32 {
            return Err("components nest too deeply (cycle?)".into());
        }
        let object = self
            .objects
            .get(&id)
            .ok_or_else(|| format!("build references missing object {id}"))?;
        if object.mesh.triangle_count() > 0 {
            out.append(&object.mesh.transformed(transform));
        }
        for c in &object.components {
            self.flatten(c.object, transform * c.transform, out, depth + 1)?;
        }
        Ok(())
    }
}

fn parse_model(xml: &str) -> std::result::Result<Model, String> {
    let mut reader = quick_xml::Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut model = Model {
        unit_to_metres: 0.001,
        ..Model::default()
    };
    let mut current: Option<(u32, ModelObject)> = None;
    let mut in_build = false;

    loop {
        let event = reader.read_event().map_err(|e| e.to_string())?;
        let (e, closes) = match &event {
            Event::Start(e) => (e, false),
            Event::Empty(e) => (e, true),
            Event::End(e) => {
                match e.local_name().as_ref() {
                    "object" => {
                        if let Some((id, o)) = current.take() {
                            model.objects.insert(id, o);
                        }
                    }
                    "build" => in_build = false,
                    _ => {}
                }
                continue;
            }
            Event::Eof => break,
            _ => continue,
        };
        let attrs = attributes(e)?;
        let get = |k: &str| attrs.get(k).map(String::as_str);
        let num = |k: &str| -> std::result::Result<f32, String> {
            get(k)
                .ok_or_else(|| format!("missing attribute {k}"))?
                .parse::<f32>()
                .map_err(|e| format!("bad number in {k}: {e}"))
        };
        match e.local_name().as_ref() {
            "model" => {
                model.unit_to_metres = match get("unit").unwrap_or("millimeter") {
                    "micron" => 1e-6,
                    "millimeter" => 1e-3,
                    "centimeter" => 1e-2,
                    "inch" => 0.0254,
                    "foot" => 0.3048,
                    "meter" => 1.0,
                    other => return Err(format!("unknown unit {other}")),
                };
            }
            "object" => {
                let id = num("id")? as u32;
                let object = ModelObject {
                    name: get("name").map(str::to_owned),
                    ..ModelObject::default()
                };
                if closes {
                    model.objects.insert(id, object);
                } else {
                    current = Some((id, object));
                }
            }
            "vertex" => {
                if let Some((_, o)) = current.as_mut() {
                    o.mesh
                        .positions
                        .push(Vec3::new(num("x")?, num("y")?, num("z")?));
                }
            }
            "triangle" => {
                if let Some((_, o)) = current.as_mut() {
                    let n = o.mesh.positions.len() as u32;
                    let idx = [num("v1")?, num("v2")?, num("v3")?].map(|v| v as u32);
                    if idx.iter().any(|&i| i >= n) {
                        return Err(format!("triangle index out of range ({n} vertices)"));
                    }
                    o.mesh.indices.extend_from_slice(&idx);
                }
            }
            "component" => {
                if let Some((_, o)) = current.as_mut() {
                    o.components.push(Item {
                        object: num("objectid")? as u32,
                        transform: parse_transform(get("transform"))?,
                    });
                }
            }
            "build" => in_build = !closes,
            "item" if in_build => model.build.push(Item {
                object: num("objectid")? as u32,
                transform: parse_transform(get("transform"))?,
            }),
            _ => {}
        }
    }
    Ok(model)
}

fn attributes(e: &BytesStart) -> std::result::Result<HashMap<String, String>, String> {
    e.attributes()
        .map(|a| {
            let a = a.map_err(|e| e.to_string())?;
            let local = a.key.local_name();
            let key: &str = local.as_ref();
            let key = key.to_owned();
            let value = a
                .normalized_value(quick_xml::XmlVersion::Implicit1_0)
                .map_err(|e| e.to_string())?
                .into_owned();
            Ok((key, value))
        })
        .collect()
}

/// Parses a 3MF transform: twelve numbers, a 4×3 matrix applied to row vectors.
fn parse_transform(attr: Option<&str>) -> std::result::Result<Mat4, String> {
    let Some(s) = attr else {
        return Ok(Mat4::IDENTITY);
    };
    let m: Vec<f32> = s
        .split_whitespace()
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| format!("bad transform: {e}"))?;
    if m.len() != 12 {
        return Err(format!("transform needs 12 numbers, found {}", m.len()));
    }
    Ok(Mat4::from_cols(
        Vec4::new(m[0], m[1], m[2], 0.0),
        Vec4::new(m[3], m[4], m[5], 0.0),
        Vec4::new(m[6], m[7], m[8], 0.0),
        Vec4::new(m[9], m[10], m[11], 1.0),
    ))
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_uses_row_vector_convention() {
        // Translate by (10, 20, 30).
        let m = parse_transform(Some("1 0 0 0 1 0 0 0 1 10 20 30")).unwrap();
        assert_eq!(m.transform_point3(Vec3::ZERO), Vec3::new(10.0, 20.0, 30.0));
        // 90° about Z: x → y.
        let r = parse_transform(Some("0 1 0 -1 0 0 0 0 1 0 0 0")).unwrap();
        assert!(r.transform_point3(Vec3::X).abs_diff_eq(Vec3::Y, 1e-6));
    }

    #[test]
    fn parses_components_and_units() {
        let xml = r#"<?xml version="1.0"?>
<model unit="centimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
<resources>
<object id="1" type="model"><mesh><vertices>
<vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/>
</vertices><triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>
<object id="2" type="model" name="pair"><components>
<component objectid="1"/><component objectid="1" transform="1 0 0 0 1 0 0 0 1 5 0 0"/>
</components></object>
</resources>
<build><item objectid="2"/></build>
</model>"#;
        let model = parse_model(xml).unwrap();
        assert!((model.unit_to_metres - 0.01).abs() < 1e-9);
        let mut mesh = Mesh::default();
        model.flatten(2, Mat4::IDENTITY, &mut mesh, 0).unwrap();
        assert_eq!(mesh.triangle_count(), 2);
        assert_eq!(mesh.positions[4], Vec3::new(6.0, 0.0, 0.0));
    }
}
