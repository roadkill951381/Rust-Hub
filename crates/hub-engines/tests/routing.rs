//! Which engine a generation request goes to.

use std::sync::{Arc, Mutex};

use hub_comfy::{ComfyUi, Config};
use hub_core::engine::{GenerateRequest, Generated, GenerationInfo, Generator, Progress};
use hub_engines::Generators;

/// Stands in for hub-gen and records what it was asked.
#[derive(Debug, Default)]
struct Recorder(Mutex<Vec<GenerateRequest>>);

impl Generator for Recorder {
    fn describe(&self) -> String {
        "recorder".into()
    }

    fn generate(&self, request: &GenerateRequest, _: Progress<'_>) -> Result<Generated, String> {
        self.0.lock().unwrap().push(request.clone());
        Ok(Generated {
            path: "unused.glb".into(),
            info: GenerationInfo::default(),
        })
    }
}

fn generators(shape: Option<&str>, image: Option<&str>) -> (Generators, Arc<Recorder>) {
    let recorder = Arc::new(Recorder::default());
    let comfy = ComfyUi::new(Config {
        // Nothing listens here, and there is no folder to start it from.
        url: "http://127.0.0.1:9".into(),
        ..Config::default()
    });
    let g = Generators {
        hub_gen: recorder.clone(),
        comfy: Arc::new(comfy),
        shape_model: shape.map(str::to_owned),
        image_model: image.map(str::to_owned),
    };
    (g, recorder)
}

fn prompt(model: Option<&str>, image_model: Option<&str>) -> GenerateRequest {
    GenerateRequest {
        prompt: Some("a stone owl".into()),
        model: model.map(str::to_owned),
        image_model: image_model.map(str::to_owned),
        ..GenerateRequest::default()
    }
}

#[test]
fn plain_requests_go_to_hub_gen_with_the_defaults() {
    let (g, recorder) = generators(Some("hunyuan3d-2mini"), None);
    g.generate(&prompt(None, None), &|_, _| {}).unwrap();
    g.generate(&prompt(Some("triposg"), Some("flux-schnell")), &|_, _| {})
        .unwrap();
    let seen = recorder.0.lock().unwrap();
    assert_eq!(seen[0].model.as_deref(), Some("hunyuan3d-2mini"));
    assert_eq!(seen[1].model.as_deref(), Some("triposg"));
    assert_eq!(seen[1].image_model.as_deref(), Some("flux-schnell"));
}

#[test]
fn comfyui_requests_go_to_comfyui() {
    let (g, recorder) = generators(None, None);
    let err = g
        .generate(&prompt(Some("comfyui"), None), &|_, _| {})
        .unwrap_err();
    assert!(err.contains("ComfyUI is not running"), "{err}");
    let err = g
        .generate(&prompt(None, Some("comfyui:flux-dev")), &|_, _| {})
        .unwrap_err();
    assert!(err.contains("ComfyUI is not running"), "{err}");
    assert!(
        recorder.0.lock().unwrap().is_empty(),
        "hub-gen was not asked"
    );

    // HUB_IMAGE_MODEL=comfyui applies to prompts, not to pictures.
    let (g, recorder) = generators(None, Some("comfyui"));
    assert!(g.generate(&prompt(None, None), &|_, _| {}).is_err());
    let picture = GenerateRequest {
        image: Some("owl.png".into()),
        ..GenerateRequest::default()
    };
    g.generate(&picture, &|_, _| {}).unwrap();
    assert_eq!(recorder.0.lock().unwrap().len(), 1);
}
