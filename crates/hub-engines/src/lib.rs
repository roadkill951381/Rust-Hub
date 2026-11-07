//! One place that decides which engines a front end gets, so the desktop app,
//! the CLI and the MCP server behave the same.
//!
//! | Engine | Found through |
//! |---|---|
//! | Blender (mesh tools, rigging) | `HUB_BLENDER`, the `PATH`, standard install folders (on first use) |
//! | Generation service | `HUB_GEN_URL` (default `http://127.0.0.1:8765`), `HUB_GEN_TOKEN` |
//! | ComfyUI | `HUB_COMFY_URL` (default `http://127.0.0.1:8188`), `HUB_COMFY_DIR` to start it |
//! | Asset library | `HUB_LIBRARY` (default `~/RustHub/library`) |
//!
//! Generation goes to [`Generators`], which sends each request to hub-gen,
//! ComfyUI, or both: ComfyUI's picture with hub-gen's shape model.
//! `HUB_SHAPE_MODEL` and `HUB_IMAGE_MODEL` set the defaults for requests that
//! name no model, e.g. `HUB_IMAGE_MODEL=comfyui` to make every picture with
//! FLUX in ComfyUI.
//!
//! Nothing is contacted or created here; each engine reports its own setup
//! problem when a command first needs it.

use std::path::PathBuf;
use std::sync::Arc;

use hub_blender::LazyBlender;
use hub_comfy::{ComfyUi, parse_engine};
use hub_core::Engines;
use hub_core::engine::{FaceRequest, GenerateRequest, Generated, Generator, Progress};
use hub_core::library::Library;
use hub_generate::Client;

/// Overrides for [`build`]; `None` fields fall back to the environment.
#[derive(Debug, Clone, Default)]
pub struct Config {
    pub library: Option<PathBuf>,
    pub generator_url: Option<String>,
}

/// Engines from the environment.
pub fn from_env() -> Engines {
    build(&Config::default())
}

pub fn build(config: &Config) -> Engines {
    let mut hub_gen = Client::from_env();
    if let Some(url) = &config.generator_url {
        hub_gen = Client::new(url.clone()).with_token(std::env::var("HUB_GEN_TOKEN").ok());
    }
    let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
    let generators = Generators {
        hub_gen: Arc::new(hub_gen),
        comfy: Arc::new(ComfyUi::from_env()),
        shape_model: var("HUB_SHAPE_MODEL"),
        image_model: var("HUB_IMAGE_MODEL"),
    };
    let library = config
        .library
        .clone()
        .or_else(Library::default_root)
        .map(Library::open);
    let blender = Arc::new(LazyBlender::new());
    Engines {
        mesh_tools: Some(blender.clone()),
        generator: Some(Arc::new(generators)),
        rigger: Some(blender),
        library,
    }
}

/// Routes generation between hub-gen and ComfyUI.
///
/// - `model: "comfyui[:workflow]"`: the whole job in ComfyUI.
/// - `image_model: "comfyui[:workflow]"` with a prompt, a `face` to keep, or a
///   `pose` guide to draw over: ComfyUI makes the picture (FLUX with PuLID for
///   a face, a turnaround sheet for `multiview`), then stops it (if the hub
///   started it) or unloads its models, so hub-gen has the GPU and memory for
///   the shape.
/// - Anything else: hub-gen.
#[derive(Debug)]
pub struct Generators {
    pub hub_gen: Arc<dyn Generator>,
    pub comfy: Arc<ComfyUi>,
    /// Shape model for requests that name none.
    pub shape_model: Option<String>,
    /// Text-to-image model for requests that name none.
    pub image_model: Option<String>,
}

impl Generator for Generators {
    fn describe(&self) -> String {
        format!("{} or {}", self.hub_gen.describe(), self.comfy.describe())
    }

    fn generate(
        &self,
        request: &GenerateRequest,
        progress: Progress<'_>,
    ) -> Result<Generated, String> {
        let mut request = request.clone();
        if request.model.is_none() {
            request.model.clone_from(&self.shape_model);
        }
        if request.image_model.is_none() {
            request.image_model.clone_from(&self.image_model);
        }
        if request.model.as_deref().and_then(parse_engine).is_some() {
            return self.comfy.generate(&request, progress);
        }
        let Picture::Comfy(workflow) = picture_workflow(&request) else {
            return self.hub_gen.generate(&request, progress);
        };
        let prompt = request
            .prompt
            .clone()
            .filter(|p| !p.trim().is_empty())
            .ok_or("give a prompt for ComfyUI to draw, or a picture")?;
        let seed = request.seed.unwrap_or_else(hub_comfy::random_seed);
        let first = |f: f32, s: &str| progress(f * 0.3, s);
        let (picture, name) = self.comfy.make_picture(
            &prompt,
            request.face.as_deref(),
            hub_comfy::pose_of(&request),
            seed,
            workflow,
            &first,
        )?;
        if self.comfy.config().free_after {
            progress(0.3, "freeing ComfyUI's memory");
            self.comfy.release();
        }
        request.image = Some(picture.clone());
        request.image_model = None;
        request.seed = Some(seed);
        let rest = |f: f32, s: &str| progress(0.3 + f * 0.7, s);
        let mut generated = self.hub_gen.generate(&request, &rest)?;
        generated.info.image_model = Some(format!("comfyui:{name}"));
        generated.info.image = Some(picture);
        Ok(generated)
    }

    /// Faces are redrawn in ComfyUI; hub-gen has no face model.
    fn redraw_face(
        &self,
        request: &FaceRequest,
        progress: Progress<'_>,
    ) -> Result<PathBuf, String> {
        self.comfy.redraw_face(request, progress)
    }
}

/// Who draws the picture for a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Picture<'a> {
    /// hub-gen, or nobody because a picture was given.
    HubGen,
    /// ComfyUI, with this workflow or the configured default.
    Comfy(Option<&'a str>),
}

/// A named workflow wins; otherwise a face picks `PuLID` and `multiview` a
/// turnaround sheet. A face or a pose guide always goes to ComfyUI: hub-gen's
/// picture models can't keep a face, and they ignore a pose guide, drawing
/// their own figure and views in any order.
fn picture_workflow(request: &GenerateRequest) -> Picture<'_> {
    if request.image.is_some() {
        return Picture::HubGen;
    }
    let named = request.image_model.as_deref().and_then(parse_engine);
    if named.is_none() && request.face.is_none() && request.pose.is_none() {
        return Picture::HubGen;
    }
    let chosen = match (request.face.is_some(), request.multiview) {
        (true, true) => Some(hub_comfy::FLUX_PULID_SHEET),
        (true, false) => Some(hub_comfy::FLUX_PULID),
        (false, true) => Some(hub_comfy::FLUX_DEV_SHEET),
        (false, false) => None,
    };
    Picture::Comfy(named.flatten().or(chosen))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(image_model: Option<&str>, face: bool, multiview: bool) -> GenerateRequest {
        GenerateRequest {
            prompt: Some("a knight".into()),
            image_model: image_model.map(Into::into),
            face: face.then(|| PathBuf::from("face.png")),
            multiview,
            ..GenerateRequest::default()
        }
    }

    #[test]
    fn faces_and_turnarounds_pick_their_comfy_workflows() {
        assert_eq!(
            picture_workflow(&request(None, false, false)),
            Picture::HubGen
        );
        assert_eq!(
            picture_workflow(&request(None, false, true)),
            Picture::HubGen
        );
        assert_eq!(
            picture_workflow(&request(Some("comfyui"), false, false)),
            Picture::Comfy(None)
        );
        assert_eq!(
            picture_workflow(&request(Some("comfyui"), false, true)),
            Picture::Comfy(Some(hub_comfy::FLUX_DEV_SHEET))
        );
        assert_eq!(
            picture_workflow(&request(None, true, false)),
            Picture::Comfy(Some(hub_comfy::FLUX_PULID))
        );
        assert_eq!(
            picture_workflow(&request(Some("hunyuan-dit"), true, true)),
            Picture::Comfy(Some(hub_comfy::FLUX_PULID_SHEET))
        );
        assert_eq!(
            picture_workflow(&request(Some("comfyui:mine"), true, true)),
            Picture::Comfy(Some("mine"))
        );
        // A pose guide (rigging like a character, without a face) needs
        // ComfyUI too, even when a hub-gen picture model is configured.
        let mut posed = request(Some("hunyuan-dit"), false, true);
        posed.pose = Some("pose.png".into());
        assert_eq!(
            picture_workflow(&posed),
            Picture::Comfy(Some(hub_comfy::FLUX_DEV_SHEET))
        );
        posed.multiview = false;
        assert_eq!(picture_workflow(&posed), Picture::Comfy(None));
        let mut given = request(Some("comfyui"), true, true);
        given.image = Some("picture.png".into());
        assert_eq!(picture_workflow(&given), Picture::HubGen);
    }
}
