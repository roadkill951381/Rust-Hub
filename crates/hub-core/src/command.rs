//! The command layer: every edit the GUI, the CLI or Claude can make.
//!
//! Commands are plain data (serde + JSON Schema), so a script, an MCP tool
//! call and a button click all produce the same value and run the same code.
//! Distances in commands are millimetres and angles are degrees, because those
//! are the units people and prompts use; the scene itself stays in metres.

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::analysis::{PrintReport, RepairSummary};
use crate::primitives::Primitive;
use crate::scene::{Material, ObjectId};

/// Which dimension `scale_to_size` measures.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SizeAxis {
    /// The largest of width, depth and height.
    #[default]
    Longest,
    /// Width, along X.
    X,
    /// Depth on the print bed, the scene's Z axis.
    Depth,
    /// Height, the scene's Y axis (Z on a printer).
    Height,
}

/// One edit or query. Serialized with an `op` tag, e.g.
/// `{"op": "scale_to_size", "size_mm": 80}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Command {
    /// Load every mesh in a .glb, .gltf, .stl or .3mf file into the scene.
    Import { path: PathBuf },
    /// Write objects to a .glb, .gltf, .stl or .3mf file (format from the extension).
    /// STL and 3MF are written in millimetres, Z-up, with transforms applied.
    Export {
        path: PathBuf,
        /// Objects to write; all objects when omitted.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
    },
    /// Add a cube, sphere or cylinder whose largest dimension is `size_mm`.
    AddPrimitive {
        shape: Primitive,
        size_mm: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Remove objects.
    Delete { objects: Vec<ObjectId> },
    /// Rename one object.
    Rename { object: ObjectId, name: String },
    /// Set parts of an object's transform. Omitted fields stay as they are.
    SetTransform {
        object: ObjectId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        position_mm: Option<[f32; 3]>,
        /// XYZ Euler angles in degrees.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rotation_deg: Option<[f32; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scale: Option<[f32; 3]>,
    },
    /// Move objects by an offset.
    Translate {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        offset_mm: [f32; 3],
    },
    /// Rotate objects about their combined centre, XYZ Euler degrees in world axes.
    Rotate {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        degrees: [f32; 3],
    },
    /// Scale objects uniformly about their combined centre so the chosen
    /// dimension of their combined bounds equals `size_mm`.
    ScaleToSize {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        size_mm: f32,
        #[serde(default)]
        axis: SizeAxis,
    },
    /// Centre objects over the origin and rest their lowest point on the floor
    /// (the print bed).
    PlaceOnFloor {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
    },
    /// Bake an object's transform into its vertices and reset the transform.
    ApplyTransform { object: ObjectId },
    /// Remove zero-area and duplicate triangles, unify winding and turn
    /// inside-out shells outward. Holes are reported, not filled.
    Repair {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
    },
    /// Change an object's material.
    SetMaterial {
        object: ObjectId,
        material: Material,
    },
    /// Undo the last edit.
    Undo,
    /// Redo the last undone edit.
    Redo,
    /// List every object with its size and triangle count.
    ListObjects,
    /// Check objects for 3D printing: watertight, outward-facing, sizes in mm.
    CheckPrintable {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
    },
}

impl Command {
    /// True for commands that change the scene and so enter the undo history.
    pub fn is_edit(&self) -> bool {
        !matches!(
            self,
            Command::Export { .. }
                | Command::Undo
                | Command::Redo
                | Command::ListObjects
                | Command::CheckPrintable { .. }
        )
    }

    /// Short label for history menus and logs.
    pub fn label(&self) -> &'static str {
        match self {
            Command::Import { .. } => "Import",
            Command::Export { .. } => "Export",
            Command::AddPrimitive { .. } => "Add primitive",
            Command::Delete { .. } => "Delete",
            Command::Rename { .. } => "Rename",
            Command::SetTransform { .. } => "Set transform",
            Command::Translate { .. } => "Move",
            Command::Rotate { .. } => "Rotate",
            Command::ScaleToSize { .. } => "Scale to size",
            Command::PlaceOnFloor { .. } => "Place on floor",
            Command::ApplyTransform { .. } => "Apply transform",
            Command::Repair { .. } => "Repair",
            Command::SetMaterial { .. } => "Set material",
            Command::Undo => "Undo",
            Command::Redo => "Redo",
            Command::ListObjects => "List objects",
            Command::CheckPrintable { .. } => "Check printable",
        }
    }
}

/// A summary of one object for listings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ObjectInfo {
    pub id: ObjectId,
    pub name: String,
    pub triangles: usize,
    pub vertices: usize,
    /// World-space width (X), depth (scene Z) and height (scene Y), in mm.
    pub size_mm: [f32; 3],
    /// World-space centre of the bounds, scene axes, in mm.
    pub center_mm: [f32; 3],
    pub position_mm: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],
}

/// What a command did. Fields that do not apply are left out of the JSON.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Outcome {
    /// One sentence describing the result.
    pub summary: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub created: Vec<ObjectId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<ObjectInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reports: Vec<PrintReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repairs: Vec<RepairSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
}

impl Outcome {
    pub fn summary(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_use_op_tags() {
        let cmd: Command = serde_json::from_str(r#"{"op":"scale_to_size","size_mm":80}"#).unwrap();
        assert_eq!(
            cmd,
            Command::ScaleToSize {
                objects: None,
                size_mm: 80.0,
                axis: SizeAxis::Longest
            }
        );
        let json = serde_json::to_string(&Command::Undo).unwrap();
        assert_eq!(json, r#"{"op":"undo"}"#);
    }

    #[test]
    fn schema_lists_every_op() {
        let schema = serde_json::to_string(&schemars::schema_for!(Command)).unwrap();
        for op in [
            "import",
            "export",
            "scale_to_size",
            "check_printable",
            "repair",
        ] {
            assert!(schema.contains(op), "{op} missing from schema");
        }
    }
}
