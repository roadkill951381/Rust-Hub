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

/// Words added to a prompt so the picture suits a shape model: one whole
/// object, centred, on a plain background.
const PICTURE_STYLE: &str = ", a single complete object, full view, centered, plain white \
                             background, soft even studio lighting, no shadows, high detail";

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
                 {FLUX_DEV} and {HUNYUAN3D_21}{folder}"
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
            return Err(format!(
                "the ComfyUI workflow {} needs a picture",
                self.name
            ));
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
    match name {
        FLUX_SCHNELL => Some(flux("flux1-schnell-Q8_0.gguf", 4, None)),
        FLUX_DEV => Some(flux("flux1-dev-Q8_0.gguf", 20, Some(3.5))),
        HUNYUAN3D_21 => Some(hunyuan3d_21()),
        _ => None,
    }
}

/// FLUX through ComfyUI-GGUF's loader, the way a 12 to 16 GB card runs it.
fn flux(unet: &str, steps: u32, guidance: Option<f32>) -> Value {
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
            "text": format!("{{{{prompt}}}}{PICTURE_STYLE}"),
            "clip": ["2", 0]
        }},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": 1024, "height": 1024, "batch_size": 1
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
}
