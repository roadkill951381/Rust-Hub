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
use crate::engine::GenerationInfo;
use crate::library::Asset;
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
    /// Remove zero-area and duplicate triangles, unify winding, fill holes and
    /// turn inside-out shells outward.
    Repair {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        /// Close holes with new triangles (default true).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fill_holes: Option<bool>,
        /// Delete separate pieces smaller than this fraction of the largest
        /// piece's volume, e.g. 0.01 for the specks AI models leave behind.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        remove_small_shells: Option<f32>,
    },
    /// Hollow solid objects for resin printing, leaving walls `wall_mm` thick.
    /// Objects must be watertight; run repair first if they are not.
    Hollow {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        wall_mm: f32,
        /// Grid spacing for the cavity; smaller is more exact and slower.
        /// Defaults to half the wall thickness.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        voxel_mm: Option<f32>,
    },
    /// Rebuild objects' surfaces on a voxel grid with Blender: closes holes,
    /// merges overlapping parts and evens out triangles. Removes UVs.
    Remesh {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        /// Voxel size; detail smaller than this is lost.
        voxel_mm: f32,
    },
    /// Reduce triangle counts with Blender, keeping shape and UVs.
    /// Give either `ratio` (0 to 1) or `triangles` (target per object).
    Decimate {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ratio: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        triangles: Option<u32>,
    },
    /// Lay out new UVs with Blender's Smart UV Project, ready for texturing.
    UvUnwrap {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        /// Faces meeting at a sharper angle than this get a seam (default 66).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        angle_deg: Option<f32>,
        /// Space between UV islands, 0 to 1 of the texture (default 0.02).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        margin: Option<f32>,
    },
    /// Generate a 3D model from a text prompt or a reference image on the
    /// local generation service (the person's own GPU), add it to the scene
    /// resting on the floor, and save it to the asset library.
    Generate {
        /// Describe one object, e.g. "a knight in plate armour, full body".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        /// A picture of a single object (PNG or JPEG) to build the model from.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image: Option<PathBuf>,
        /// Shape model: "hunyuan3d-2", "hunyuan3d-2mini" or "triposg";
        /// the service's default when omitted. "comfyui" or
        /// "comfyui:<workflow>" runs the whole job in ComfyUI.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        /// Text-to-image model for a prompt: "hunyuan-dit", "flux-schnell",
        /// or "comfyui" / "comfyui:<workflow>" to make the picture in ComfyUI.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_model: Option<String>,
        /// Also paint a colour texture (slower, needs more GPU memory).
        #[serde(default)]
        texture: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
        /// Target triangle count.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        faces: Option<u32>,
        /// Name for the object and the library asset.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Scale so the longest side is this many mm.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        size_mm: Option<f32>,
        /// A portrait (PNG or JPEG) whose face the character keeps; the
        /// picture is drawn in ComfyUI with PuLID. With `rig_like`, a sharp
        /// close-up of the face is then painted onto the texture too (see
        /// `SharpenFace`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        face: Option<PathBuf>,
        /// Draw the front, side and back, and build the shape from all of
        /// them (Hunyuan3D-2mv): truer backs, hair and profiles.
        #[serde(default)]
        multiview: bool,
        /// A rigged character (.glb) to rig the result like: the same
        /// skeleton, rest pose and animations, so a game can swap it in. The
        /// rigged file is the one saved to the library.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rig_like: Option<PathBuf>,
        /// Start of the material names when rigging, e.g. `surv_mei` for
        /// `surv_mei.body` and `surv_mei.clothes`; the reference's by default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        materials: Option<String>,
        /// When rigging, draw over the reference as it looks (its colours and
        /// clothes) instead of a plain grey figure: for remaking that same
        /// character. Needs `rig_like`.
        #[serde(default)]
        keep_look: bool,
        /// How much of the pose guide the picture repaints, from 0 to 1:
        /// lower keeps more of the reference's pose (and look). 0.88 by
        /// default, 0.7 with `keep_look`. Needs `rig_like`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repaint: Option<f32>,
    },
    /// Rig a model (.glb) like an existing rigged character (.glb) with
    /// Blender: the character's skeleton, rest pose and animations are kept
    /// exactly and its meshes are swapped for the model, skinned to the same
    /// bones. Writes `output` and adds the result to the scene to look at.
    Rig {
        model: PathBuf,
        reference: PathBuf,
        output: PathBuf,
        /// Reduce the model to about this many triangles first.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        triangles: Option<u32>,
        /// Start of the material names, e.g. `surv_mei`; the reference's by default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        materials: Option<String>,
    },
    /// Paint a sharp face onto a rigged, textured character (.glb, as `Rig`
    /// writes it): Blender renders its head close up, ComfyUI redraws that
    /// render with the likeness of `face` (FLUX with PuLID), and Blender
    /// paints the result back onto the texture. Writes `output` and adds it
    /// to the scene to look at.
    SharpenFace {
        model: PathBuf,
        /// The portrait (PNG or JPEG) whose face to keep.
        face: PathBuf,
        output: PathBuf,
        /// Who it is, e.g. the prompt the character was made from.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt: Option<String>,
        /// How much of the render to redraw, from 0 to 1: more is sharper
        /// and more like the portrait, less keeps the eyes and mouth exactly
        /// where the model has them. 0.65 by default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repaint: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        seed: Option<u64>,
    },
    /// Bring a rigged, textured character (.glb, as `Rig` writes it) to a
    /// game character's colour levels: its skin and its clothes are each
    /// darkened or lightened, keeping their hues, to the brightness of the
    /// same parts of `like` (generated textures are often much lighter than
    /// the albedo a game's lights expect). Can also shrink the texture.
    /// Writes `output` and adds it to the scene to look at.
    MatchColours {
        model: PathBuf,
        /// The game's own character (.glb), usually the one the model was
        /// rigged like.
        like: PathBuf,
        output: PathBuf,
        /// How far to go, from 0 to 1: 1 (the default) matches the
        /// brightness, less keeps more of the model's own.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strength: Option<f32>,
        /// Shrink the texture to at most this many pixels a side.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_texture: Option<u32>,
        /// Write the texture as a JPEG (quality 90), several times smaller
        /// than a PNG; for textures without transparency.
        #[serde(default)]
        jpeg: bool,
    },
    /// Build a character kit for an in-game character customizer from a kit
    /// recipe: for each base body (such as male and female) a rigged body
    /// with every slider (weight, muscle, jaw, nose...) baked in as morph
    /// targets, and the hair, outfits and other parts that go on it, rigged
    /// to the same skeleton and following the same sliders. Writes
    /// `kit.json` and the GLBs into `output`, and adds the bodies to the
    /// scene to look at. Needs Blender with the MPFB extension.
    BuildCharacterKit {
        recipe: PathBuf,
        /// The kit folder.
        output: PathBuf,
        /// Build only these bodies, keeping the rest of an existing kit.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bodies: Option<Vec<String>>,
    },
    /// Save objects as a new version of a library asset.
    SaveAsset {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        note: Option<String>,
    },
    /// Add a library asset to the scene, the latest version unless one is given.
    LoadAsset {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        version: Option<u32>,
    },
    /// List library assets, optionally only those whose name contains `query`.
    ListAssets {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query: Option<String>,
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
    /// Check objects for 3D printing: watertight, outward-facing, wall
    /// thickness, sizes in mm.
    CheckPrintable {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        objects: Option<Vec<ObjectId>>,
        /// Thinnest wall to accept, in mm (default 0.8, two 0.4 mm perimeters).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min_wall_mm: Option<f32>,
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
                | Command::SaveAsset { .. }
                | Command::ListAssets { .. }
        )
    }

    /// True for commands that call an engine or take long enough that a GUI
    /// should run them off its main thread with [`Session::begin`](crate::Session::begin).
    pub fn is_job(&self) -> bool {
        matches!(
            self,
            Command::Hollow { .. }
                | Command::Remesh { .. }
                | Command::Decimate { .. }
                | Command::UvUnwrap { .. }
                | Command::Generate { .. }
                | Command::Rig { .. }
                | Command::SharpenFace { .. }
                | Command::MatchColours { .. }
                | Command::BuildCharacterKit { .. }
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
            Command::Hollow { .. } => "Hollow",
            Command::Remesh { .. } => "Remesh",
            Command::Decimate { .. } => "Decimate",
            Command::UvUnwrap { .. } => "UV unwrap",
            Command::Generate { .. } => "Generate",
            Command::Rig { .. } => "Rig",
            Command::SharpenFace { .. } => "Sharpen face",
            Command::MatchColours { .. } => "Match colours",
            Command::BuildCharacterKit { .. } => "Build character kit",
            Command::SaveAsset { .. } => "Save asset",
            Command::LoadAsset { .. } => "Load asset",
            Command::ListAssets { .. } => "List assets",
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<Asset>,
    /// The library version this command wrote.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved: Option<SavedAsset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationInfo>,
}

/// A version written to the asset library.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SavedAsset {
    pub name: String,
    pub version: u32,
    pub path: PathBuf,
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
            "hollow",
            "generate",
            "save_asset",
        ] {
            assert!(schema.contains(op), "{op} missing from schema");
        }
    }
}
