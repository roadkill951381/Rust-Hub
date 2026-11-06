//! Rust-Hub MCP server.
//!
//! Every tool here builds a [`hub_core::Command`] and runs it through the same
//! [`Session::execute`] the desktop app uses, so Claude and a person at the
//! GUI share one code path. Tool results are the command's JSON outcome.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use base64::Engine as _;
use hub_core::primitives::Primitive;
use hub_core::{Command, Material, ObjectId, Session, SizeAxis, Step};
use hub_render::{FrameOptions, Headless};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, ProgressNotificationParam, ServerCapabilities,
    ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

const INSTRUCTIONS: &str = "\
Rust-Hub edits 3D models for games and 3D printing.

Units and axes: tool arguments use millimetres and degrees. The scene is Y-up \
(glTF); STL and 3MF files are written in millimetres with Z up, and sizes in \
reports are width (X), depth and height. Object ids come from import, \
add_primitive and list_objects.

Typical print workflow: import -> scale_to_size -> place_on_floor -> \
check_printable -> repair if needed -> export to .stl or .3mf. Call \
render_preview to look at the result. Every edit can be undone with undo.

Making new models: generate (a text prompt or a picture) runs on the person's \
own GPU through the local hub-gen service, adds the model resting on the floor \
and saves it to the asset library. For printing, pass size_mm, then \
check_printable; generated models often need repair with \
remove_small_shells 0.01. Use hollow for resin prints, decimate to cut \
triangle counts for games, remesh to fuse overlapping parts, and uv_unwrap \
before texturing (remesh, decimate and uv_unwrap run Blender). \
check_printable also measures wall thickness against min_wall_mm (0.8 by default).

Game characters: generate with face (a portrait to keep the likeness of), \
multiview true, a prompt describing the full figure in the reference's rest \
pose (\"full body, A-pose, arms 45 degrees away from the body\"), and rig_like \
set to the game's rigged character .glb. The result keeps that character's \
skeleton and animations, so the game can load it in place of the old model. \
rig does the same for a model made earlier.";

/// The MCP server state: one editing session and a lazily created renderer.
#[derive(Clone)]
pub struct HubServer {
    session: Arc<Mutex<Session>>,
    renderer: Arc<Mutex<Option<Headless>>>,
    tool_router: ToolRouter<Self>,
}

impl std::fmt::Debug for HubServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HubServer").finish_non_exhaustive()
    }
}

impl Default for HubServer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct PathArgs {
    /// Absolute path of a .glb, .gltf, .stl or .3mf file.
    pub path: PathBuf,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportArgs {
    /// Output path; the extension (.glb, .gltf, .stl, .3mf) picks the format.
    pub path: PathBuf,
    /// Object ids to write; every object when omitted.
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddPrimitiveArgs {
    pub shape: Primitive,
    /// Largest dimension in millimetres.
    pub size_mm: f32,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SelectionArgs {
    /// Object ids; every object when omitted.
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ObjectsArgs {
    pub objects: Vec<ObjectId>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ObjectArgs {
    pub object: ObjectId,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RenameArgs {
    pub object: ObjectId,
    pub name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetTransformArgs {
    pub object: ObjectId,
    /// Position in millimetres, scene axes (Y up).
    #[serde(default)]
    pub position_mm: Option<[f32; 3]>,
    /// XYZ Euler angles in degrees.
    #[serde(default)]
    pub rotation_deg: Option<[f32; 3]>,
    /// Per-axis scale factors.
    #[serde(default)]
    pub scale: Option<[f32; 3]>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TranslateArgs {
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// Offset in millimetres, scene axes (Y up).
    pub offset_mm: [f32; 3],
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RotateArgs {
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// XYZ Euler angles in degrees, about the selection's centre.
    pub degrees: [f32; 3],
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScaleToSizeArgs {
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// Target size in millimetres.
    pub size_mm: f32,
    /// Which dimension to measure; the longest side by default.
    #[serde(default)]
    pub axis: SizeAxis,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CheckArgs {
    /// Object ids; every object when omitted.
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// Thinnest acceptable wall in mm (default 0.8, two 0.4 mm nozzle lines).
    #[serde(default)]
    pub min_wall_mm: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RepairArgs {
    /// Object ids; every object when omitted.
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// Close holes with new triangles (default true).
    #[serde(default)]
    pub fill_holes: Option<bool>,
    /// Delete loose pieces smaller than this fraction of the largest piece's
    /// volume; 0.01 suits AI-generated models.
    #[serde(default)]
    pub remove_small_shells: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct HollowArgs {
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// Wall thickness in mm; 2 is typical for resin.
    pub wall_mm: f32,
    /// Grid spacing in mm; defaults to half the wall.
    #[serde(default)]
    pub voxel_mm: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RemeshArgs {
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// Voxel size in mm; detail smaller than this is lost.
    pub voxel_mm: f32,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DecimateArgs {
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// Fraction of triangles to keep, 0 to 1.
    #[serde(default)]
    pub ratio: Option<f32>,
    /// Target triangle count per object (instead of ratio).
    #[serde(default)]
    pub triangles: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UnwrapArgs {
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// Seam angle in degrees (default 66).
    #[serde(default)]
    pub angle_deg: Option<f32>,
    /// Island margin, 0 to 0.5 of the texture (default 0.02).
    #[serde(default)]
    pub margin: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateArgs {
    /// One object to make, e.g. "a knight in plate armour, full body, standing".
    #[serde(default)]
    pub prompt: Option<String>,
    /// Absolute path of a PNG or JPEG of one object, used instead of a prompt.
    #[serde(default)]
    pub image: Option<PathBuf>,
    /// Shape model: hunyuan3d-2 (default), hunyuan3d-2mini or triposg on the
    /// hub-gen service, or "comfyui" / "comfyui:<workflow>" to run the whole
    /// job in the person's ComfyUI.
    #[serde(default)]
    pub model: Option<String>,
    /// Text-to-image model for a prompt: hunyuan-dit (default) or
    /// flux-schnell on hub-gen, or "comfyui" to make the picture with the
    /// person's ComfyUI (FLUX) and the shape on hub-gen.
    #[serde(default)]
    pub image_model: Option<String>,
    /// Also paint a colour texture (slower).
    #[serde(default)]
    pub texture: bool,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Target triangle count.
    #[serde(default)]
    pub faces: Option<u32>,
    /// Name for the object and its library asset.
    #[serde(default)]
    pub name: Option<String>,
    /// Scale so the longest side is this many mm.
    #[serde(default)]
    pub size_mm: Option<f32>,
    /// Absolute path of a portrait whose face the character should keep
    /// (drawn in ComfyUI with PuLID). Use with a prompt describing the
    /// whole figure, outfit and pose.
    #[serde(default)]
    pub face: Option<PathBuf>,
    /// Draw front, side and back views and build the shape from all of them
    /// (Hunyuan3D-2mv): better backs, hair and profiles for characters.
    #[serde(default)]
    pub multiview: bool,
    /// Absolute path of a rigged character (.glb) to rig the result like:
    /// same skeleton, rest pose and animations, so a game can swap it in.
    /// Implies a texture. The prompt should ask for that character's rest
    /// pose, usually an A-pose.
    #[serde(default)]
    pub rig_like: Option<PathBuf>,
    /// Start of material names when rigging, e.g. `surv_mei` gives
    /// `surv_mei.body` and `surv_mei.clothes`; the reference's by default.
    #[serde(default)]
    pub materials: Option<String>,
    /// With `rig_like`: draw over the reference as it looks (colours,
    /// clothes) instead of a plain grey figure, to remake that same
    /// character with a new face or details. Repaints 0.7 by default.
    #[serde(default)]
    pub keep_look: bool,
    /// With `rig_like`: how much of the reference the picture repaints, 0
    /// to 1 (default 0.88, or 0.7 with `keep_look`). Lower keeps the pose,
    /// arms and colours better; higher follows the prompt and face more.
    #[serde(default)]
    pub repaint: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RigArgs {
    /// Absolute path of the model to rig (.glb), e.g. a generated character.
    pub model: PathBuf,
    /// Absolute path of a rigged character (.glb) whose skeleton, rest pose
    /// and animations to keep.
    pub reference: PathBuf,
    /// Absolute path to write the rigged model (.glb).
    pub output: PathBuf,
    /// Reduce the model to about this many triangles first.
    #[serde(default)]
    pub triangles: Option<u32>,
    /// Start of the material names, e.g. `surv_mei`; the reference's by default.
    #[serde(default)]
    pub materials: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SharpenFaceArgs {
    /// Absolute path of a rigged, textured character (.glb), e.g. from rig.
    pub model: PathBuf,
    /// Absolute path of the portrait (PNG or JPEG) whose face to keep.
    pub face: PathBuf,
    /// Absolute path to write the result (.glb).
    pub output: PathBuf,
    /// Who it is, e.g. the prompt the character was made from.
    #[serde(default)]
    pub prompt: Option<String>,
    /// How much of the close-up to redraw, 0 to 1 (0.55 by default): more is
    /// sharper and more like the portrait, less keeps the eyes and mouth
    /// exactly where the model has them.
    #[serde(default)]
    pub repaint: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SaveAssetArgs {
    pub name: String,
    #[serde(default)]
    pub objects: Option<Vec<ObjectId>>,
    /// What changed in this version.
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoadAssetArgs {
    pub name: String,
    /// Version number; the latest when omitted.
    #[serde(default)]
    pub version: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListAssetsArgs {
    /// Only assets whose name contains this text.
    #[serde(default)]
    pub query: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetMaterialArgs {
    pub object: ObjectId,
    pub material: Material,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunCommandsArgs {
    /// Commands to run in order; stops at the first failure.
    pub commands: Vec<Command>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RenderArgs {
    /// Image width in pixels (default 768).
    #[serde(default)]
    pub width: Option<u32>,
    /// Image height in pixels (default 576).
    #[serde(default)]
    pub height: Option<u32>,
    /// Orbit angle around the vertical axis in degrees (default 35).
    #[serde(default)]
    pub yaw_deg: Option<f32>,
    /// Elevation in degrees, positive looks down (default 25).
    #[serde(default)]
    pub pitch_deg: Option<f32>,
    /// Object to highlight.
    #[serde(default)]
    pub highlight: Option<ObjectId>,
}

#[tool_router]
impl HubServer {
    /// A server with the engines found in the environment (see `hub-engines`).
    pub fn new() -> Self {
        Self::with_session(Session::with_engines(hub_engines::from_env()))
    }

    /// A server around an existing session, e.g. one with test engines.
    pub fn with_session(session: Session) -> Self {
        Self {
            session: Arc::new(Mutex::new(session)),
            renderer: Arc::new(Mutex::new(None)),
            tool_router: Self::tool_router(),
        }
    }

    /// Runs a long command without holding the session while it works, so
    /// other tools stay responsive, and reports progress to the client when
    /// it asked for it.
    async fn run_job(
        &self,
        command: Command,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let session = Arc::clone(&self.session);
        let step = tokio::task::spawn_blocking(move || lock(&session).begin(command))
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let job = match step {
            Ok(Step::Done(outcome)) => return Ok(success(&outcome)),
            Ok(Step::Job(job)) => job,
            Err(e) => {
                return Ok(CallToolResult::error(vec![ContentBlock::text(
                    e.to_string(),
                )]));
            }
        };

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(f32, String)>();
        let work = tokio::task::spawn_blocking(move || {
            job.run(&move |fraction, stage| {
                let _ = tx.send((fraction, stage.to_owned()));
            })
        });
        if let Some(token) = context.meta.get_progress_token() {
            let peer = context.peer.clone();
            tokio::spawn(async move {
                let mut last = String::new();
                while let Some((fraction, stage)) = rx.recv().await {
                    if stage == last && fraction < 1.0 {
                        continue;
                    }
                    last.clone_from(&stage);
                    let param = ProgressNotificationParam::new(token.clone(), f64::from(fraction))
                        .with_total(1.0)
                        .with_message(stage);
                    let _ = peer.notify_progress(param).await;
                }
            });
        }
        let output = work
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let session = Arc::clone(&self.session);
        let result = tokio::task::spawn_blocking(move || {
            output.and_then(|output| lock(&session).finish(output))
        })
        .await
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(match result {
            Ok(outcome) => success(&outcome),
            Err(e) => CallToolResult::error(vec![ContentBlock::text(e.to_string())]),
        })
    }

    /// Runs a command on a blocking thread and turns the outcome into a tool result.
    async fn run(&self, command: Command) -> Result<CallToolResult, McpError> {
        let session = Arc::clone(&self.session);
        let result = tokio::task::spawn_blocking(move || lock(&session).execute(command))
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(match result {
            Ok(outcome) => success(&outcome),
            Err(e) => CallToolResult::error(vec![ContentBlock::text(e.to_string())]),
        })
    }

    #[tool(
        description = "Load every mesh in a .glb, .gltf, .stl or .3mf file into the scene. Returns the new object ids and sizes."
    )]
    async fn import(
        &self,
        Parameters(a): Parameters<PathArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::Import { path: a.path }).await
    }

    #[tool(
        description = "Write objects to a file; the extension picks the format. STL and 3MF are millimetres, Z-up, ready for a slicer. GLB/glTF keep transforms and materials for game engines such as three.js."
    )]
    async fn export(
        &self,
        Parameters(a): Parameters<ExportArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::Export {
            path: a.path,
            objects: a.objects,
        })
        .await
    }

    #[tool(description = "Add a cube, sphere or cylinder whose largest dimension is size_mm.")]
    async fn add_primitive(
        &self,
        Parameters(a): Parameters<AddPrimitiveArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::AddPrimitive {
            shape: a.shape,
            size_mm: a.size_mm,
            name: a.name,
        })
        .await
    }

    #[tool(description = "List every object with id, name, triangle count, size and transform.")]
    async fn list_objects(&self) -> Result<CallToolResult, McpError> {
        self.run(Command::ListObjects).await
    }

    #[tool(description = "Remove objects from the scene.")]
    async fn delete(
        &self,
        Parameters(a): Parameters<ObjectsArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::Delete { objects: a.objects }).await
    }

    #[tool(description = "Rename an object.")]
    async fn rename(
        &self,
        Parameters(a): Parameters<RenameArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::Rename {
            object: a.object,
            name: a.name,
        })
        .await
    }

    #[tool(
        description = "Set an object's position (mm), rotation (degrees) and/or scale. Omitted fields keep their values."
    )]
    async fn set_transform(
        &self,
        Parameters(a): Parameters<SetTransformArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::SetTransform {
            object: a.object,
            position_mm: a.position_mm,
            rotation_deg: a.rotation_deg,
            scale: a.scale,
        })
        .await
    }

    #[tool(description = "Move objects by an offset in millimetres.")]
    async fn translate(
        &self,
        Parameters(a): Parameters<TranslateArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::Translate {
            objects: a.objects,
            offset_mm: a.offset_mm,
        })
        .await
    }

    #[tool(
        description = "Rotate objects about their combined centre by XYZ Euler angles in degrees."
    )]
    async fn rotate(
        &self,
        Parameters(a): Parameters<RotateArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::Rotate {
            objects: a.objects,
            degrees: a.degrees,
        })
        .await
    }

    #[tool(
        description = "Scale objects uniformly so one dimension of their combined bounds equals size_mm. axis is longest (default), x, depth or height."
    )]
    async fn scale_to_size(
        &self,
        Parameters(a): Parameters<ScaleToSizeArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::ScaleToSize {
            objects: a.objects,
            size_mm: a.size_mm,
            axis: a.axis,
        })
        .await
    }

    #[tool(
        description = "Centre objects over the origin and rest their lowest point on the floor (the print bed)."
    )]
    async fn place_on_floor(
        &self,
        Parameters(a): Parameters<SelectionArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::PlaceOnFloor { objects: a.objects }).await
    }

    #[tool(description = "Bake an object's transform into its vertices and reset the transform.")]
    async fn apply_transform(
        &self,
        Parameters(a): Parameters<ObjectArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::ApplyTransform { object: a.object }).await
    }

    #[tool(
        description = "Check objects for 3D printing: watertight (no holes), outward-facing, no zero-area faces, wall thickness against min_wall_mm, with size (mm), volume, issues (block printing) and warnings (e.g. thin walls)."
    )]
    async fn check_printable(
        &self,
        Parameters(a): Parameters<CheckArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::CheckPrintable {
            objects: a.objects,
            min_wall_mm: a.min_wall_mm,
        })
        .await
    }

    #[tool(
        description = "Fix common mesh problems: remove zero-area and duplicate triangles, unify face winding, fill holes, flip inside-out shells, and optionally drop small loose pieces. Returns a fresh print check."
    )]
    async fn repair(
        &self,
        Parameters(a): Parameters<RepairArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::Repair {
            objects: a.objects,
            fill_holes: a.fill_holes,
            remove_small_shells: a.remove_small_shells,
        })
        .await
    }

    #[tool(
        description = "Hollow watertight objects for resin printing, leaving walls wall_mm thick. Returns a print check of the result."
    )]
    async fn hollow(
        &self,
        Parameters(a): Parameters<HollowArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let command = Command::Hollow {
            objects: a.objects,
            wall_mm: a.wall_mm,
            voxel_mm: a.voxel_mm,
        };
        self.run_job(command, context).await
    }

    #[tool(
        description = "Rebuild surfaces on a voxel grid with Blender: closes holes, fuses overlapping parts, evens out triangles. Removes UVs."
    )]
    async fn remesh(
        &self,
        Parameters(a): Parameters<RemeshArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let command = Command::Remesh {
            objects: a.objects,
            voxel_mm: a.voxel_mm,
        };
        self.run_job(command, context).await
    }

    #[tool(
        description = "Reduce triangle counts with Blender, keeping shape and UVs. Give ratio (0 to 1) or triangles."
    )]
    async fn decimate(
        &self,
        Parameters(a): Parameters<DecimateArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let command = Command::Decimate {
            objects: a.objects,
            ratio: a.ratio,
            triangles: a.triangles,
        };
        self.run_job(command, context).await
    }

    #[tool(description = "Lay out new UVs with Blender's Smart UV Project, ready for texturing.")]
    async fn uv_unwrap(
        &self,
        Parameters(a): Parameters<UnwrapArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let command = Command::UvUnwrap {
            objects: a.objects,
            angle_deg: a.angle_deg,
            margin: a.margin,
        };
        self.run_job(command, context).await
    }

    #[tool(
        description = "Generate a 3D model from a text prompt or a picture on the person's own GPU (local hub-gen service; nothing is billed). Adds it resting on the floor, optionally scaled to size_mm, and saves it to the asset library. Takes from seconds to a few minutes; the first run downloads the model."
    )]
    async fn generate(
        &self,
        Parameters(a): Parameters<GenerateArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let command = Command::Generate {
            prompt: a.prompt,
            image: a.image,
            model: a.model,
            image_model: a.image_model,
            texture: a.texture,
            seed: a.seed,
            faces: a.faces,
            name: a.name,
            size_mm: a.size_mm,
            face: a.face,
            multiview: a.multiview,
            rig_like: a.rig_like,
            materials: a.materials,
            keep_look: a.keep_look,
            repaint: a.repaint,
        };
        self.run_job(command, context).await
    }

    #[tool(
        description = "Rig a model (.glb) like an existing rigged character (.glb) with Blender. The character's skeleton, rest pose and animations are kept exactly, and its meshes are swapped for the model, skinned to the same bones and split into skin and clothes materials, so a game can load it in place of the old one. Writes output and adds the result to the scene. The model should stand in the reference's rest pose (usually an A-pose); arm angles that differ are corrected."
    )]
    async fn rig(
        &self,
        Parameters(a): Parameters<RigArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let command = Command::Rig {
            model: a.model,
            reference: a.reference,
            output: a.output,
            triangles: a.triangles,
            materials: a.materials,
        };
        self.run_job(command, context).await
    }

    #[tool(
        description = "Paint a sharp face onto a rigged, textured character (.glb, as rig writes it). Blender renders the head close up, ComfyUI redraws that render with the face in a portrait (FLUX with PuLID), and Blender paints it back onto the texture, so the face is crisp and like the portrait while the skeleton, animations and the rest of the texture stay as they are. generate does this by itself when given both face and rig_like. Writes output and adds it to the scene."
    )]
    async fn sharpen_face(
        &self,
        Parameters(a): Parameters<SharpenFaceArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let command = Command::SharpenFace {
            model: a.model,
            face: a.face,
            output: a.output,
            prompt: a.prompt,
            repaint: a.repaint,
            seed: a.seed,
        };
        self.run_job(command, context).await
    }

    #[tool(description = "Save objects as a new version of a named asset in the library.")]
    async fn save_asset(
        &self,
        Parameters(a): Parameters<SaveAssetArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::SaveAsset {
            name: a.name,
            objects: a.objects,
            note: a.note,
        })
        .await
    }

    #[tool(
        description = "Add a library asset to the scene, the latest version unless one is given."
    )]
    async fn load_asset(
        &self,
        Parameters(a): Parameters<LoadAssetArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::LoadAsset {
            name: a.name,
            version: a.version,
        })
        .await
    }

    #[tool(
        description = "List library assets with every version: how each was made (prompt, model, seed), size and note."
    )]
    async fn list_assets(
        &self,
        Parameters(a): Parameters<ListAssetsArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::ListAssets { query: a.query }).await
    }

    #[tool(
        description = "Set an object's PBR material: linear RGBA base_color (0 to 1), metallic and roughness."
    )]
    async fn set_material(
        &self,
        Parameters(a): Parameters<SetMaterialArgs>,
    ) -> Result<CallToolResult, McpError> {
        self.run(Command::SetMaterial {
            object: a.object,
            material: a.material,
        })
        .await
    }

    #[tool(description = "Undo the last edit.")]
    async fn undo(&self) -> Result<CallToolResult, McpError> {
        self.run(Command::Undo).await
    }

    #[tool(description = "Redo the last undone edit.")]
    async fn redo(&self) -> Result<CallToolResult, McpError> {
        self.run(Command::Redo).await
    }

    #[tool(
        description = "Run several commands in order, each an object with an \"op\" field, e.g. {\"op\":\"scale_to_size\",\"size_mm\":80}. Stops at the first failure and reports every outcome so far."
    )]
    async fn run_commands(
        &self,
        Parameters(a): Parameters<RunCommandsArgs>,
    ) -> Result<CallToolResult, McpError> {
        let session = Arc::clone(&self.session);
        let (outcomes, failure) = tokio::task::spawn_blocking(move || {
            let mut session = session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut outcomes = Vec::new();
            for (i, command) in a.commands.into_iter().enumerate() {
                match session.execute(command) {
                    Ok(o) => outcomes.push(o),
                    Err(e) => return (outcomes, Some(format!("command {i} failed: {e}"))),
                }
            }
            (outcomes, None)
        })
        .await
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let body = serde_json::json!({ "outcomes": outcomes, "error": failure });
        let text = ContentBlock::text(serde_json::to_string_pretty(&body).unwrap_or_default());
        Ok(if failure.is_some() {
            CallToolResult::error(vec![text])
        } else {
            CallToolResult::success(vec![text])
        })
    }

    #[tool(
        description = "Check every piece Rust-Hub runs on for newer versions: the hub itself, hub-gen's Python packages, model weights, ComfyUI with its custom nodes and model files, Blender, uv, Rust and the NVIDIA driver. Changes nothing. Each item has a status (current, available, held_back, not_installed, unknown) and, when it can move forward, the exact upgrade commands with their folders. Ask the person before running any of them: some download gigabytes, and hub-gen or ComfyUI must be stopped first when `blocked` says so."
    )]
    async fn check_updates(&self) -> Result<CallToolResult, McpError> {
        let report =
            tokio::task::spawn_blocking(|| hub_updates::check(&hub_updates::Setup::from_env()))
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let body = serde_json::json!({ "summary": report.summary(), "pieces": report.pieces });
        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&body).unwrap_or_default(),
        )]))
    }

    #[tool(
        description = "Render the scene to a PNG so you can see it. Back faces show red, which reveals holes and flipped faces. The grid spacing is a power of ten in metres."
    )]
    async fn render_preview(
        &self,
        Parameters(a): Parameters<RenderArgs>,
    ) -> Result<CallToolResult, McpError> {
        let session = Arc::clone(&self.session);
        let renderer = Arc::clone(&self.renderer);
        let result = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
            let scene = session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .scene()
                .clone();
            if scene.is_empty() {
                return Err("the scene is empty".into());
            }
            let mut guard = renderer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if guard.is_none() {
                *guard = Some(Headless::new().map_err(|e| e.to_string())?);
            }
            let headless = guard.as_mut().expect("renderer initialised above");
            let mut camera = hub_render::framing_camera(&scene);
            if let Some(yaw) = a.yaw_deg {
                camera.yaw = yaw.to_radians();
            }
            if let Some(pitch) = a.pitch_deg {
                camera.pitch = pitch.clamp(-89.0, 89.0).to_radians();
            }
            headless
                .render_png(
                    &scene,
                    &camera,
                    a.width.unwrap_or(768).clamp(64, 2048),
                    a.height.unwrap_or(576).clamp(64, 2048),
                    FrameOptions {
                        selected: a.highlight,
                        hide_grid: false,
                    },
                )
                .map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(match result {
            Ok(png) => CallToolResult::success(vec![ContentBlock::image(
                base64::engine::general_purpose::STANDARD.encode(png),
                "image/png",
            )]),
            Err(e) => {
                CallToolResult::error(vec![ContentBlock::text(format!("render failed: {e}"))])
            }
        })
    }
}

// `tool_handler` generates `async fn call_tool` without an `.await`.
#[allow(clippy::unused_async_trait_impl)]
#[tool_handler(router = self.tool_router)]
impl ServerHandler for HubServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("rust-hub", env!("CARGO_PKG_VERSION"))
                    .with_title("Rust-Hub 3D tools"),
            )
            .with_instructions(INSTRUCTIONS)
    }
}

fn lock(session: &Mutex<Session>) -> std::sync::MutexGuard<'_, Session> {
    session
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn success(outcome: &hub_core::Outcome) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(
        serde_json::to_string_pretty(outcome).unwrap_or_else(|_| outcome.summary.clone()),
    )])
}

/// Serves the hub over stdin/stdout until the client disconnects.
pub async fn serve_stdio() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use rmcp::ServiceExt as _;
    let service = HubServer::new().serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
