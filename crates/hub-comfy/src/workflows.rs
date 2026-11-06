//! Workflows in ComfyUI's API format, with `{{placeholders}}` the hub fills in.
//!
//! The built-in ones use ComfyUI's own nodes plus ComfyUI-GGUF for FLUX, and
//! the model file names those projects publish. A person's own workflow
//! (exported with "Export (API)") in the workflows folder replaces a built-in
//! one of the same name, or adds a new one.
//!
//! Placeholders, anywhere in a string input:
//!
//! | Placeholder | Becomes |
//! |---|---|
//! | `{{prompt}}` | The text prompt |
//! | `{{image}}` | The uploaded picture's name, for a `LoadImage` node |
//! | `{{seed}}` | The seed; a string that is exactly `"{{seed}}"` becomes a number |

use std::path::Path;

use serde_json::{Value, json};

/// Picture from a prompt with FLUX.1-schnell: 4 steps, fast.
pub const FLUX_SCHNELL: &str = "flux-schnell";
/// Picture from a prompt with FLUX.1-dev: 20 steps, better detail.
pub const FLUX_DEV: &str = "flux-dev";
/// Shape from a picture with ComfyUI's built-in Hunyuan3D 2.1 nodes.
pub const HUNYUAN3D_21: &str = "hunyuan3d-2.1";
/// A turnaround sheet (front, side facing left, back) with FLUX.1-dev, for
/// multiview shape models.
pub const FLUX_DEV_SHEET: &str = "flux-dev-sheet";
/// A full-length character with the face of `{{image}}`: FLUX.1-dev with
/// PuLID (`ComfyUI_PuLID_Flux_ll`), a smaller FLUX so both fit 12 GB.
pub const FLUX_PULID: &str = "flux-pulid";
/// [`FLUX_PULID`] as a turnaround sheet.
pub const FLUX_PULID_SHEET: &str = "flux-pulid-sheet";

/// Words added to a prompt so the picture suits a shape model: one whole
/// object, centred, on a plain background.
const PICTURE_STYLE: &str = ", a single complete object, full view, centered, plain white \
                             background, soft even studio lighting, no shadows, high detail";

/// Words for a whole standing figure that a rig can be fitted to.
const FIGURE_STYLE: &str = "full-length shot of the whole body from head to toes, feet \
                            fully visible, empty space above the head and below the feet, \
                            standing straight in an A-pose with the arms held away from the \
                            body and the hands open, plain light grey background, soft even \
                            studio lighting, no shadows, high detail";

/// Words for a turnaround sheet that multiview shape models split into views.
const SHEET_STYLE: &str = "character turnaround reference sheet, the same character shown \
                           three times side by side, evenly spaced and not overlapping: \
                           front view, side view facing left, back view";

/// What a workflow makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Makes {
    Picture,
    Model,
}

/// A workflow ready to fill in.
#[derive(Debug, Clone, PartialEq)]
pub struct Workflow {
    pub name: String,
    pub graph: Value,
    /// Read from the person's workflows folder rather than built in.
    pub custom: bool,
}

impl Workflow {
    /// Loads `name`: `<folder>/<name>.json` if it exists, else a built-in one.
    pub fn load(name: &str, folder: Option<&Path>) -> Result<Self, String> {
        if let Some(folder) = folder {
            let path = folder.join(format!("{name}.json"));
            if path.is_file() {
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("could not read {}: {e}", path.display()))?;
                let graph: Value = serde_json::from_str(&text)
                    .map_err(|e| format!("{} is not valid JSON: {e}", path.display()))?;
                check_api_format(&graph, &path)?;
                return Ok(Self {
                    name: name.to_owned(),
                    graph,
                    custom: true,
                });
            }
        }
        let graph = builtin(name).ok_or_else(|| {
            let folder = folder.map_or_else(String::new, |f| {
                format!(
                    ", and there is no {}",
                    f.join(format!("{name}.json")).display()
                )
            });
            format!(
                "no ComfyUI workflow called {name:?}: the built-in ones are {FLUX_SCHNELL}, \
                 {FLUX_DEV}, {FLUX_DEV_SHEET}, {FLUX_PULID}, {FLUX_PULID_SHEET} and \
                 {HUNYUAN3D_21}{folder}"
            )
        })?;
        Ok(Self {
            name: name.to_owned(),
            graph,
            custom: false,
        })
    }

    pub fn uses(&self, placeholder: &str) -> bool {
        fn walk(v: &Value, needle: &str) -> bool {
            match v {
                Value::String(s) => s.contains(needle),
                Value::Array(items) => items.iter().any(|i| walk(i, needle)),
                Value::Object(map) => map.values().any(|i| walk(i, needle)),
                _ => false,
            }
        }
        walk(&self.graph, placeholder)
    }

    /// Whether the workflow saves a 3D model (a `SaveGLB` node, or any node
    /// whose type mentions GLB or 3D saving) or only pictures.
    pub fn makes(&self) -> Makes {
        let saves_model = self.graph.as_object().is_some_and(|nodes| {
            nodes.values().any(|n| {
                n.get("class_type")
                    .and_then(Value::as_str)
                    .is_some_and(|t| {
                        let t = t.to_ascii_lowercase();
                        t.contains("glb") || (t.contains("save") && t.contains("3d"))
                    })
            })
        });
        if saves_model {
            Makes::Model
        } else {
            Makes::Picture
        }
    }

    /// The graph with every placeholder replaced.
    pub fn fill(
        &self,
        prompt: Option<&str>,
        image: Option<&str>,
        seed: u64,
    ) -> Result<Value, String> {
        fn walk(v: &mut Value, prompt: &str, image: &str, seed: u64) {
            match v {
                Value::String(s) if s == "{{seed}}" => *v = Value::from(seed),
                Value::String(s) => {
                    if s.contains("{{") {
                        *s = s
                            .replace("{{prompt}}", prompt)
                            .replace("{{image}}", image)
                            .replace("{{seed}}", &seed.to_string());
                    }
                }
                Value::Array(items) => items.iter_mut().for_each(|i| walk(i, prompt, image, seed)),
                Value::Object(map) => map.values_mut().for_each(|i| walk(i, prompt, image, seed)),
                _ => {}
            }
        }
        if self.uses("{{prompt}}") && prompt.is_none() {
            return Err(format!("the ComfyUI workflow {} needs a prompt", self.name));
        }
        if self.uses("{{image}}") && image.is_none() {
            return Err(if self.uses("{{prompt}}") {
                format!(
                    "the ComfyUI workflow {} needs a face picture (a portrait) as well as \
                     the prompt",
                    self.name
                )
            } else {
                format!("the ComfyUI workflow {} needs a picture", self.name)
            });
        }
        let mut graph = self.graph.clone();
        walk(&mut graph, prompt.unwrap_or(""), image.unwrap_or(""), seed);
        Ok(graph)
    }
}

/// API-format workflows map node ids to `{class_type, inputs}`; the editor's
/// own format has a `nodes` list instead and cannot be queued.
fn check_api_format(graph: &Value, path: &Path) -> Result<(), String> {
    let api = graph.as_object().is_some_and(|nodes| {
        !nodes.is_empty() && nodes.values().all(|n| n.get("class_type").is_some())
    });
    if api {
        Ok(())
    } else {
        Err(format!(
            "{} is not in ComfyUI's API format. In ComfyUI use Workflow > Export (API) \
             and save that file instead",
            path.display()
        ))
    }
}

fn builtin(name: &str) -> Option<Value> {
    let object = Shot {
        style: PICTURE_STYLE.to_owned(),
        width: 1024,
        height: 1024,
    };
    let figure = Shot {
        style: format!(", {FIGURE_STYLE}"),
        width: 832,
        height: 1216,
    };
    let sheet = Shot {
        style: format!(", {SHEET_STYLE}, each {FIGURE_STYLE}"),
        width: 1536,
        height: 768,
    };
    match name {
        FLUX_SCHNELL => Some(flux("flux1-schnell-Q8_0.gguf", 4, None, &object)),
        FLUX_DEV => Some(flux("flux1-dev-Q8_0.gguf", 20, Some(3.5), &object)),
        FLUX_DEV_SHEET => Some(flux("flux1-dev-Q8_0.gguf", 24, Some(3.5), &sheet)),
        FLUX_PULID => Some(with_pulid(flux(PULID_UNET, 24, Some(3.5), &figure))),
        FLUX_PULID_SHEET => Some(with_pulid(flux(PULID_UNET, 24, Some(3.5), &sheet))),
        HUNYUAN3D_21 => Some(hunyuan3d_21()),
        _ => None,
    }
}

/// The text after the prompt and the picture's size.
struct Shot {
    style: String,
    width: u32,
    height: u32,
}

/// FLUX.1-dev small enough to share 12 GB with PuLID's encoders.
const PULID_UNET: &str = "flux1-dev-Q5_K_S.gguf";

/// A model file a built-in workflow loads, and where it comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile {
    pub name: &'static str,
    /// Folders under `ComfyUI/models` that ComfyUI looks in, preferred first.
    pub folders: &'static [&'static str],
    /// The Hugging Face repo it downloads from, at the repo's top level.
    pub repo: &'static str,
    /// The repo needs a sign-in and accepting its licence on the website.
    pub gated: bool,
    /// The built-in workflows that load it.
    pub workflows: &'static [&'static str],
}

const UNET: &[&str] = &["unet", "diffusion_models"];
const TEXT: &[&str] = &["text_encoders", "clip"];
const FLUXES: &[&str] = &[
    FLUX_SCHNELL,
    FLUX_DEV,
    FLUX_DEV_SHEET,
    FLUX_PULID,
    FLUX_PULID_SHEET,
];
const PULIDS: &[&str] = &[FLUX_PULID, FLUX_PULID_SHEET];

/// Every file the built-in workflows load.
pub const MODEL_FILES: &[ModelFile] = &[
    ModelFile {
        name: "flux1-schnell-Q8_0.gguf",
        folders: UNET,
        repo: "city96/FLUX.1-schnell-gguf",
        gated: false,
        workflows: &[FLUX_SCHNELL],
    },
    ModelFile {
        name: "flux1-dev-Q8_0.gguf",
        folders: UNET,
        repo: "city96/FLUX.1-dev-gguf",
        gated: false,
        workflows: &[FLUX_DEV, FLUX_DEV_SHEET],
    },
    ModelFile {
        name: PULID_UNET,
        folders: UNET,
        repo: "city96/FLUX.1-dev-gguf",
        gated: false,
        workflows: PULIDS,
    },
    ModelFile {
        name: "pulid_flux_v0.9.1.safetensors",
        folders: &["pulid"],
        repo: "guozinan/PuLID",
        gated: false,
        workflows: PULIDS,
    },
    ModelFile {
        name: "t5xxl_fp8_e4m3fn.safetensors",
        folders: TEXT,
        repo: "comfyanonymous/flux_text_encoders",
        gated: false,
        workflows: FLUXES,
    },
    ModelFile {
        name: "clip_l.safetensors",
        folders: TEXT,
        repo: "comfyanonymous/flux_text_encoders",
        gated: false,
        workflows: FLUXES,
    },
    ModelFile {
        name: "ae.safetensors",
        folders: &["vae"],
        repo: "black-forest-labs/FLUX.1-schnell",
        gated: true,
        workflows: FLUXES,
    },
    ModelFile {
        name: "hunyuan_3d_v2.1.safetensors",
        folders: &["checkpoints"],
        repo: "Comfy-Org/hunyuan3D_2.1_repackaged",
        gated: false,
        workflows: &[HUNYUAN3D_21],
    },
];

/// FLUX through ComfyUI-GGUF's loader, the way a 12 to 16 GB card runs it.
fn flux(unet: &str, steps: u32, guidance: Option<f32>, shot: &Shot) -> Value {
    let positive = if guidance.is_some() {
        json!(["10", 0])
    } else {
        json!(["4", 0])
    };
    let mut graph = json!({
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux"
        }},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {
            "text": format!("{{{{prompt}}}}{}", shot.style),
            "clip": ["2", 0]
        }},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": shot.width, "height": shot.height, "batch_size": 1
        }},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0],
            "seed": "{{seed}}",
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": positive,
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "denoise": 1.0
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {
            "images": ["8", 0], "filename_prefix": "rust-hub/picture"
        }}
    });
    if let Some(g) = guidance {
        graph["10"] = json!({"class_type": "FluxGuidance", "inputs": {
            "conditioning": ["4", 0], "guidance": g
        }});
    }
    graph
}

/// Puts PuLID between the FLUX model and the sampler, so the picture keeps
/// the face in `{{image}}` (`ComfyUI_PuLID_Flux_ll` nodes; its EVA-CLIP and
/// InsightFace models download on first use).
fn with_pulid(mut graph: Value) -> Value {
    graph["20"] = json!({"class_type": "PulidFluxModelLoader", "inputs": {
        "pulid_file": "pulid_flux_v0.9.1.safetensors"
    }});
    graph["21"] = json!({"class_type": "PulidFluxEvaClipLoader", "inputs": {}});
    graph["22"] =
        json!({"class_type": "PulidFluxInsightFaceLoader", "inputs": {"provider": "CUDA"}});
    graph["23"] = json!({"class_type": "LoadImage", "inputs": {"image": "{{image}}"}});
    graph["24"] = json!({"class_type": "ApplyPulidFlux", "inputs": {
        "model": ["1", 0],
        "pulid_flux": ["20", 0],
        "eva_clip": ["21", 0],
        "face_analysis": ["22", 0],
        "image": ["23", 0],
        "weight": 1.0,
        "start_at": 0.0,
        "end_at": 1.0,
        "fusion": "mean",
        "fusion_weight_max": 1.0,
        "fusion_weight_min": 0.0,
        "train_step": 1000,
        "use_gray": true
    }});
    graph["7"]["inputs"]["model"] = json!(["24", 0]);
    graph
}

/// ComfyUI's own "Image to Model (Hunyuan3D 2.1)" template, unpacked.
fn hunyuan3d_21() -> Value {
    json!({
        "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {
            "ckpt_name": "hunyuan_3d_v2.1.safetensors"
        }},
        "2": {"class_type": "LoadImage", "inputs": {"image": "{{image}}"}},
        "3": {"class_type": "CLIPVisionEncode", "inputs": {
            "clip_vision": ["1", 1], "image": ["2", 0], "crop": "center"
        }},
        "4": {"class_type": "Hunyuan3Dv2Conditioning", "inputs": {"clip_vision_output": ["3", 0]}},
        "5": {"class_type": "EmptyLatentHunyuan3Dv2", "inputs": {"resolution": 4096, "batch_size": 1}},
        "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 1.0}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["6", 0],
            "seed": "{{seed}}",
            "steps": 30,
            "cfg": 5.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "positive": ["4", 0],
            "negative": ["4", 1],
            "latent_image": ["5", 0],
            "denoise": 1.0
        }},
        "8": {"class_type": "VAEDecodeHunyuan3D", "inputs": {
            "samples": ["7", 0], "vae": ["1", 2], "num_chunks": 8000, "octree_resolution": 256
        }},
        "9": {"class_type": "VoxelToMesh", "inputs": {
            "voxel": ["8", 0], "algorithm": "surface net", "threshold": 0.6
        }},
        "10": {"class_type": "SaveGLB", "inputs": {"mesh": ["9", 0], "filename_prefix": "rust-hub/model"}}
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_fill_in_and_say_what_they_make() {
        let flux = Workflow::load(FLUX_SCHNELL, None).unwrap();
        assert_eq!(flux.makes(), Makes::Picture);
        let g = flux.fill(Some("a stone owl"), None, 42).unwrap();
        assert_eq!(g["7"]["inputs"]["seed"], json!(42));
        let text = g["4"]["inputs"]["text"].as_str().unwrap();
        assert!(
            text.starts_with("a stone owl, a single complete object"),
            "{text}"
        );
        assert!(flux.fill(None, None, 1).is_err());

        let dev = Workflow::load(FLUX_DEV, None).unwrap();
        let g = dev.fill(Some("x"), None, 1).unwrap();
        assert_eq!(g["7"]["inputs"]["positive"], json!(["10", 0]));
        assert_eq!(g["7"]["inputs"]["steps"], json!(20));

        let shape = Workflow::load(HUNYUAN3D_21, None).unwrap();
        assert_eq!(shape.makes(), Makes::Model);
        assert!(shape.fill(None, None, 1).unwrap_err().contains("picture"));
        let g = shape.fill(None, Some("owl.png"), 7).unwrap();
        assert_eq!(g["2"]["inputs"]["image"], json!("owl.png"));
    }

    #[test]
    fn a_persons_workflow_replaces_a_builtin() {
        let dir = tempfile::tempdir().unwrap();
        let mine =
            json!({"1": {"class_type": "SaveImage", "inputs": {"filename_prefix": "{{prompt}}"}}});
        std::fs::write(dir.path().join("flux-schnell.json"), mine.to_string()).unwrap();
        let w = Workflow::load(FLUX_SCHNELL, Some(dir.path())).unwrap();
        assert!(w.custom);
        assert_eq!(
            w.fill(Some("p"), None, 1).unwrap()["1"]["inputs"]["filename_prefix"],
            json!("p")
        );

        std::fs::write(dir.path().join("ui.json"), r#"{"nodes": [], "links": []}"#).unwrap();
        let err = Workflow::load("ui", Some(dir.path())).unwrap_err();
        assert!(err.contains("Export (API)"), "{err}");
        let err = Workflow::load("nope", Some(dir.path())).unwrap_err();
        assert!(err.contains("nope.json"), "{err}");
    }

    #[test]
    fn character_workflows_take_a_face_and_draw_whole_figures() {
        let pulid = Workflow::load(FLUX_PULID, None).unwrap();
        assert_eq!(pulid.makes(), Makes::Picture);
        let err = pulid.fill(Some("Mei"), None, 1).unwrap_err();
        assert!(err.contains("face picture"), "{err}");
        let g = pulid.fill(Some("Mei"), Some("mei.png"), 12).unwrap();
        assert_eq!(g["23"]["inputs"]["image"], json!("mei.png"));
        assert_eq!(g["7"]["inputs"]["model"], json!(["24", 0]));
        assert!(
            g["4"]["inputs"]["text"]
                .as_str()
                .unwrap()
                .contains("A-pose")
        );

        let sheet = Workflow::load(FLUX_PULID_SHEET, None).unwrap();
        let g = sheet.fill(Some("Mei"), Some("mei.png"), 12).unwrap();
        assert_eq!(g["6"]["inputs"]["width"], json!(1536));
        let text = g["4"]["inputs"]["text"].as_str().unwrap();
        assert!(
            text.contains("front view, side view facing left, back view"),
            "{text}"
        );
        assert!(
            !Workflow::load(FLUX_DEV_SHEET, None)
                .unwrap()
                .uses("{{image}}")
        );
    }

    #[test]
    fn every_file_a_builtin_loads_is_listed() {
        for name in [
            FLUX_SCHNELL,
            FLUX_DEV,
            FLUX_DEV_SHEET,
            FLUX_PULID,
            FLUX_PULID_SHEET,
            HUNYUAN3D_21,
        ] {
            let graph = builtin(name).unwrap().to_string();
            let listed: Vec<_> = MODEL_FILES
                .iter()
                .filter(|f| f.workflows.contains(&name))
                .collect();
            for f in &listed {
                assert!(graph.contains(f.name), "{name} does not load {}", f.name);
            }
            let loads = graph.matches(".gguf").count() + graph.matches(".safetensors").count();
            assert_eq!(
                loads,
                listed.len(),
                "{name} loads a file MODEL_FILES misses"
            );
        }
    }
}
