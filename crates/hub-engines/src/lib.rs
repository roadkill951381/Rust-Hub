//! One place that decides which engines a front end gets, so the desktop app,
//! the CLI and the MCP server behave the same.
//!
//! | Engine | Found through |
//! |---|---|
//! | Blender | `HUB_BLENDER`, the `PATH`, standard install folders (on first use) |
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
use hub_core::engine::{GenerateRequest, Generated, Generator, Progress};
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
    Engines {
        mesh_tools: Some(Arc::new(LazyBlender::new())),
        generator: Some(Arc::new(generators)),
        library,
    }
}

/// Routes generation between hub-gen and ComfyUI.
///
/// - `model: "comfyui[:workflow]"`: the whole job in ComfyUI.
/// - `image_model: "comfyui[:workflow]"` with a prompt: ComfyUI makes the
///   picture, then stops it (if the hub started it) or unloads its models,
///   so hub-gen has the GPU and memory for the shape.
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
        let comfy_picture = request
            .image_model
            .as_deref()
            .and_then(parse_engine)
            .filter(|_| request.image.is_none());
        let Some(workflow) = comfy_picture else {
            return self.hub_gen.generate(&request, progress);
        };
        let prompt = request
            .prompt
            .clone()
            .filter(|p| !p.trim().is_empty())
            .ok_or("give a prompt for ComfyUI to draw, or a picture")?;
        let seed = request.seed.unwrap_or_else(hub_comfy::random_seed);
        let first = |f: f32, s: &str| progress(f * 0.3, s);
        let (picture, name) = self.comfy.make_picture(&prompt, seed, workflow, &first)?;
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
}
