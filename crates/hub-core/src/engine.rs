//! Engines: the external tools the hub drives, behind traits so `hub-core`
//! stays free of process, network and GPU code.
//!
//! - [`MeshTools`] runs heavy mesh operations, implemented by `hub-blender`
//!   with Blender in the background.
//! - [`Generator`] makes models from a prompt or image, implemented by
//!   `hub-gen` against the local generation service on the person's own GPU.
//! - [`Library`](crate::library::Library) keeps saved models and their versions.
//!
//! Front ends build an [`Engines`] once and hand it to
//! [`Session::with_engines`](crate::Session::with_engines). Commands that need a
//! missing engine fail with a message that says how to set it up.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::library::Library;
use crate::mesh::Mesh;

/// A mesh operation for [`MeshTools`]. Sizes are in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeshOp {
    /// Rebuild the surface on a voxel grid: closes holes, merges overlapping
    /// parts and evens out the triangles. Loses UVs.
    VoxelRemesh { voxel: f32 },
    /// Reduce the triangle count to `ratio` (0 to 1) of the current one,
    /// keeping the shape and UVs as well as possible.
    Decimate { ratio: f32 },
    /// Cut seams and lay out new UVs, so the model can be textured.
    UvUnwrap { angle_deg: f32, margin: f32 },
}

impl MeshOp {
    pub fn label(&self) -> &'static str {
        match self {
            MeshOp::VoxelRemesh { .. } => "Remesh",
            MeshOp::Decimate { .. } => "Decimate",
            MeshOp::UvUnwrap { .. } => "UV unwrap",
        }
    }
}

/// Runs mesh operations in an external tool.
pub trait MeshTools: Send + Sync + fmt::Debug {
    /// Name and version for messages, e.g. "Blender 5.2.3".
    fn describe(&self) -> String;

    /// Applies `op` to a world-space mesh and returns the new world-space mesh.
    fn apply(&self, mesh: &Mesh, op: MeshOp) -> Result<Mesh, String>;
}

/// What to generate. Mirrors [`Command::Generate`](crate::Command::Generate).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GenerateRequest {
    /// Describe the object, e.g. "a knight in plate armour, full body, standing".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// A reference picture (PNG or JPEG) of a single object; used instead of
    /// making one from the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<PathBuf>,
    /// Shape model, such as "hunyuan3d-2", "hunyuan3d-2mini" or "triposg";
    /// the service's default when omitted. "comfyui" (or "comfyui:<workflow>")
    /// runs the whole job in ComfyUI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Text-to-image model that turns the prompt into the picture the shape is
    /// made from, such as "hunyuan-dit" or "flux-schnell"; "comfyui" (or
    /// "comfyui:<workflow>") makes it in ComfyUI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_model: Option<String>,
    /// Paint a colour texture onto the shape (slower, needs more GPU memory).
    #[serde(default)]
    pub texture: bool,
    /// Random seed, for repeatable results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Target triangle count for the result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<u32>,
}

/// Where a generated model came from, stored with it in the library.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GenerationInfo {
    pub model: String,
    /// The text-to-image model, when a picture was made from the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    pub seed: u64,
    pub textured: bool,
    /// Wall-clock time on the generation service, in seconds.
    pub seconds: f32,
    /// The intermediate picture the shape was made from, when one was made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<PathBuf>,
}

/// A finished generation: a GLB on disk plus how it was made.
#[derive(Debug, Clone, PartialEq)]
pub struct Generated {
    pub path: PathBuf,
    pub info: GenerationInfo,
}

/// Progress of a long job: fraction done (0 to 1) and the current stage.
pub type Progress<'a> = &'a (dyn Fn(f32, &str) + Sync);

/// Makes 3D models.
pub trait Generator: Send + Sync + fmt::Debug {
    /// Where the generator runs, for messages, e.g. `http://127.0.0.1:8765`.
    fn describe(&self) -> String;

    /// Generates one model, reporting progress along the way. Blocks until done.
    fn generate(
        &self,
        request: &GenerateRequest,
        progress: Progress<'_>,
    ) -> Result<Generated, String>;
}

/// The engines available to a session. Cheap to clone.
#[derive(Debug, Clone, Default)]
pub struct Engines {
    pub mesh_tools: Option<Arc<dyn MeshTools>>,
    pub generator: Option<Arc<dyn Generator>>,
    pub library: Option<Library>,
}
