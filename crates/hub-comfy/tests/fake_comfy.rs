//! Drives [`ComfyUi`] against a small stand-in for ComfyUI's HTTP API that
//! answers the way ComfyUI does: uploads, queueing, history and downloads.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use hub_comfy::{ComfyUi, Config};
use hub_core::engine::{GenerateRequest, Generator};
use hub_core::scene::{Material, NewObject, Scene, Transform};
use hub_core::{io, primitives};
use serde_json::{Value, json};

#[derive(Default)]
struct State {
    uploads: Vec<String>,
    queued: Vec<Value>,
    history: HashMap<String, Value>,
    files: HashMap<String, Vec<u8>>,
    freed: usize,
    /// Checkpoints ComfyUI "has"; others are refused like a missing file.
    checkpoints: Vec<String>,
}

struct Fake {
    url: String,
    state: Arc<Mutex<State>>,
}

fn glb_bytes(dir: &std::path::Path) -> Vec<u8> {
    let mut scene = Scene::default();
    let id = scene.add(NewObject {
        name: "owl".into(),
        mesh: primitives::cube(1.0),
        transform: Transform::IDENTITY,
        material: Material::default(),
    });
    let path = dir.join("model.glb");
    io::export(&path, &[scene.get(id).unwrap()]).unwrap();
    std::fs::read(path).unwrap()
}

fn start(checkpoints: &[&str]) -> Fake {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let url = format!("http://{}", server.server_addr().to_ip().unwrap());
    let state = Arc::new(Mutex::new(State {
        checkpoints: checkpoints.iter().map(|s| (*s).to_owned()).collect(),
        ..State::default()
    }));
    let shared = Arc::clone(&state);
    let scratch = tempfile::tempdir().unwrap().keep();
    let glb = glb_bytes(&scratch);
    std::thread::spawn(move || {
        for mut request in server.incoming_requests() {
            let mut body = Vec::new();
            let _ = request.as_reader().read_to_end(&mut body);
            let url = request.url().to_owned();
            let (path, query) = url.split_once('?').unwrap_or((&url, ""));
            let mut s = shared.lock().unwrap();
            let reply = |code: u16, v: Value| {
                tiny_http::Response::from_string(v.to_string())
                    .with_status_code(code)
                    .with_header(
                        "Content-Type: application/json"
                            .parse::<tiny_http::Header>()
                            .unwrap(),
                    )
            };
            let response = match path {
                "/system_stats" => {
                    // Unloading models gives memory back, as on a real GPU.
                    let free = 2_000_000_000_u64 + 6_000_000_000 * s.freed as u64;
                    reply(200, json!({"devices": [{"vram_free": free}]}))
                }
                "/free" => {
                    s.freed += 1;
                    reply(200, json!({}))
                }
                "/upload/image" => {
                    let text = String::from_utf8_lossy(&body);
                    let name = text
                        .split("filename=\"")
                        .nth(1)
                        .and_then(|r| r.split('"').next())
                        .unwrap_or("upload.png")
                        .to_owned();
                    s.uploads.push(name.clone());
                    reply(200, json!({"name": name, "subfolder": "", "type": "input"}))
                }
                "/prompt" => {
                    let job: Value = serde_json::from_slice(&body).unwrap();
                    let graph = job["prompt"].clone();
                    let missing = graph.as_object().unwrap().iter().find(|(_, n)| {
                        n["class_type"] == "ImageOnlyCheckpointLoader"
                            && !s
                                .checkpoints
                                .iter()
                                .any(|c| n["inputs"]["ckpt_name"] == c.as_str())
                    });
                    if let Some((id, n)) = missing {
                        reply(
                            400,
                            json!({
                                "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                                "node_errors": {id.clone(): {"class_type": "ImageOnlyCheckpointLoader", "errors": [{
                                    "message": "Value not in list",
                                    "details": format!("ckpt_name: '{}' not in []", n["inputs"]["ckpt_name"].as_str().unwrap())
                                }]}}
                            }),
                        )
                    } else {
                        let id = format!("job-{}", s.queued.len());
                        let mut outputs = serde_json::Map::new();
                        for (node, n) in graph.as_object().unwrap() {
                            match n["class_type"].as_str().unwrap() {
                                "SaveImage" => {
                                    let name = format!("{id}.png");
                                    s.files.insert(name.clone(), b"\x89PNG fake".to_vec());
                                    outputs.insert(node.clone(), json!({"images": [
                                        {"filename": name, "subfolder": "rust-hub", "type": "output"}
                                    ]}));
                                }
                                "SaveGLB" => {
                                    let name = format!("{id}.glb");
                                    s.files.insert(name.clone(), glb.clone());
                                    outputs.insert(node.clone(), json!({"3d": [
                                        {"filename": name, "subfolder": "rust-hub", "type": "output"}
                                    ]}));
                                }
                                _ => {}
                            }
                        }
                        s.history.insert(id.clone(), json!({
                            "outputs": outputs,
                            "status": {"status_str": "success", "completed": true, "messages": []}
                        }));
                        s.queued.push(graph);
                        reply(
                            200,
                            json!({"prompt_id": id, "number": 1, "node_errors": {}}),
                        )
                    }
                }
                p if p.starts_with("/history/") => {
                    let id = &p["/history/".len()..];
                    match s.history.get(id) {
                        Some(entry) => reply(200, json!({id: entry})),
                        None => reply(200, json!({})),
                    }
                }
                "/view" => {
                    let name = query
                        .split('&')
                        .find_map(|kv| kv.strip_prefix("filename="))
                        .unwrap_or("");
                    match s.files.get(name) {
                        Some(bytes) => {
                            tiny_http::Response::from_data(bytes.clone()).with_status_code(200)
                        }
                        None => reply(404, json!({})),
                    }
                }
                _ => reply(404, json!({})),
            };
            drop(s);
            let _ = request.respond(response);
        }
    });
    Fake { url, state }
}

fn client(fake: &Fake, cache: PathBuf) -> ComfyUi {
    ComfyUi::new(Config {
        url: fake.url.clone(),
        ..Config::default()
    })
    .with_cache_dir(cache)
    .with_poll_interval(Duration::from_millis(10))
}

#[test]
fn a_prompt_becomes_a_picture_then_a_model() {
    let fake = start(&["hunyuan_3d_v2.1.safetensors"]);
    let cache = tempfile::tempdir().unwrap();
    let comfy = client(&fake, cache.path().to_owned());
    assert!(comfy.is_running());

    let request = GenerateRequest {
        prompt: Some("a stone owl statue".into()),
        model: Some("comfyui".into()),
        seed: Some(5),
        ..GenerateRequest::default()
    };
    let generated = comfy.generate(&request, &|_, _| {}).unwrap();
    assert!(generated.path.extension().is_some_and(|e| e == "glb"));
    assert!(io::import(&generated.path).is_ok());
    assert_eq!(generated.info.model, "comfyui:hunyuan3d-2.1");
    assert_eq!(
        generated.info.image_model.as_deref(),
        Some("comfyui:flux-schnell")
    );
    assert!(generated.info.image.as_ref().is_some_and(|p| p.is_file()));

    let s = fake.state.lock().unwrap();
    assert_eq!(
        s.queued.len(),
        2,
        "a picture workflow, then a model workflow"
    );
    let picture = &s.queued[0];
    assert_eq!(picture["7"]["inputs"]["seed"], json!(5));
    assert!(
        picture["4"]["inputs"]["text"]
            .as_str()
            .unwrap()
            .starts_with("a stone owl statue")
    );
    let model = &s.queued[1];
    assert_eq!(s.uploads.len(), 1);
    assert_eq!(model["2"]["inputs"]["image"], json!(s.uploads[0]));
    assert_eq!(s.freed, 1, "models unloaded after the job");
}

#[test]
fn a_picture_alone_feeds_the_model_workflow() {
    let fake = start(&["hunyuan_3d_v2.1.safetensors"]);
    let dir = tempfile::tempdir().unwrap();
    let comfy = client(&fake, dir.path().join("cache"));
    let picture = dir.path().join("owl.png");
    std::fs::write(&picture, b"\x89PNG fake").unwrap();
    let (path, workflow) = comfy.make_picture("x", 1, None, &|_, _| {}).unwrap();
    assert!(path.is_file());
    assert_eq!(workflow, "flux-schnell");

    let generated = comfy
        .generate(
            &GenerateRequest {
                image: Some(picture),
                model: Some("comfyui".into()),
                ..GenerateRequest::default()
            },
            &|_, _| {},
        )
        .unwrap();
    assert_eq!(generated.info.image_model, None);
    assert_eq!(fake.state.lock().unwrap().queued.len(), 2);
}

#[test]
fn a_missing_model_file_is_explained() {
    let fake = start(&[]);
    let cache = tempfile::tempdir().unwrap();
    let comfy = client(&fake, cache.path().to_owned());
    let err = comfy
        .generate(
            &GenerateRequest {
                prompt: Some("owl".into()),
                model: Some("comfyui".into()),
                ..GenerateRequest::default()
            },
            &|_, _| {},
        )
        .unwrap_err();
    assert!(err.contains("hunyuan_3d_v2.1.safetensors"), "{err}");
    assert!(
        err.contains("missing from ComfyUI's models folder"),
        "{err}"
    );
}

#[test]
fn a_stopped_comfyui_without_a_folder_says_how_to_fix_it() {
    let comfy = ComfyUi::new(Config {
        url: "http://127.0.0.1:9".into(),
        ..Config::default()
    });
    assert!(!comfy.is_running());
    let err = comfy.ensure_running(&|_, _| {}).unwrap_err();
    assert!(err.contains("HUB_COMFY_DIR"), "{err}");
}
