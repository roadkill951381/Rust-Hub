//! The phase 2 gate, from MCP tool calls only: a text prompt becomes a
//! printable model, with nothing billed outside the person's machine.
//!
//! Uses a stand-in generator that returns the kind of mesh AI models produce
//! (a hole and a floating speck). Set `HUB_GEN_TEST_URL` (and
//! `HUB_GEN_TEST_TOKEN` if the service has one) to run it against a real
//! `hub-gen serve` instead, e.g. one started with `--demo`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use hub_core::engine::{GenerateRequest, Generated, GenerationInfo, Generator, Progress};
use hub_core::library::Library;
use hub_core::scene::{Material, NewObject, Scene, Transform};
use hub_core::{Engines, Mesh, Session, io, primitives};
use hub_mcp::HubServer;
use rmcp::ServiceExt as _;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use serde_json::{Value, json};

async fn call(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    name: &'static str,
    args: Value,
) -> Value {
    let arguments = args.as_object().cloned().unwrap_or_default();
    let result: CallToolResult = client
        .call_tool(CallToolRequestParams::new(name).with_arguments(arguments))
        .await
        .unwrap_or_else(|e| panic!("{name} transport error: {e}"));
    let text = &result.content[0].as_text().expect("text result").text;
    let body = serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()));
    assert_ne!(result.is_error, Some(true), "{name} failed: {body}");
    body
}

/// Stands in for the GPU service: a sphere with one face missing and a
/// small cube floating next to it.
#[derive(Debug)]
struct RoughGenerator(PathBuf);

impl Generator for RoughGenerator {
    fn describe(&self) -> String {
        "test generator".into()
    }

    fn generate(
        &self,
        request: &GenerateRequest,
        progress: Progress<'_>,
    ) -> Result<Generated, String> {
        progress(0.5, "shape");
        let mut mesh = primitives::uv_sphere(0.5, 32, 16);
        mesh.indices.truncate(mesh.indices.len() - 3 * 40);
        let speck = primitives::cube(0.01).transformed(glam::Mat4::from_translation(
            glam::Vec3::new(0.0, 0.0, 0.62),
        ));
        append(&mut mesh, &speck);
        let mut scene = Scene::default();
        let id = scene.add(NewObject {
            name: "mesh".into(),
            mesh,
            transform: Transform::IDENTITY,
            material: Material::default(),
        });
        let path = self.0.join("generated.glb");
        io::export(&path, &[scene.get(id).unwrap()]).map_err(|e| e.to_string())?;
        Ok(Generated {
            path,
            info: GenerationInfo {
                model: "test".into(),
                prompt: request.prompt.clone(),
                seed: request.seed.unwrap_or(1),
                textured: false,
                seconds: 0.1,
                image: None,
            },
        })
    }
}

fn append(mesh: &mut Mesh, other: &Mesh) {
    let base = u32::try_from(mesh.positions.len()).unwrap();
    mesh.positions.extend_from_slice(&other.positions);
    mesh.indices.extend(other.indices.iter().map(|i| i + base));
}

fn generator(scratch: &Path) -> Arc<dyn Generator> {
    match std::env::var("HUB_GEN_TEST_URL") {
        Ok(url) => Arc::new(
            hub_generate::Client::new(url)
                .with_token(std::env::var("HUB_GEN_TEST_TOKEN").ok())
                .with_cache_dir(scratch.join("cache")),
        ),
        Err(_) => Arc::new(RoughGenerator(scratch.to_owned())),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn claude_turns_a_prompt_into_a_printable_model() {
    let dir = tempfile::tempdir().unwrap();
    let engines = Engines {
        mesh_tools: None,
        generator: Some(generator(dir.path())),
        library: Some(Library::open(dir.path().join("library"))),
    };
    let (server_io, client_io) = tokio::io::duplex(1 << 20);
    let server = tokio::spawn(async move {
        HubServer::with_session(Session::with_engines(engines))
            .serve(server_io)
            .await
            .expect("server starts")
            .waiting()
            .await
    });
    let client = ().serve(client_io).await.expect("client connects");

    // One prompt in…
    let r = call(
        &client,
        "generate",
        json!({"prompt": "a round stone with a hole", "model": "demo", "size_mm": 60, "seed": 3}),
    )
    .await;
    let id = r["created"][0]
        .as_u64()
        .expect("generate returns the new object");
    assert_eq!(r["saved"]["version"], json!(1), "{r}");
    let size = &r["objects"][0]["size_mm"];
    let longest = (0..3)
        .map(|i| size[i].as_f64().unwrap())
        .fold(0.0, f64::max);
    assert!((longest - 60.0).abs() < 0.01, "{size}");

    // …checked, repaired, hollowed and checked again…
    call(&client, "check_printable", json!({"objects": [id]})).await;
    let r = call(
        &client,
        "repair",
        json!({"objects": [id], "remove_small_shells": 0.01}),
    )
    .await;
    assert!(r["repairs"].is_array(), "{r}");
    let r = call(&client, "check_printable", json!({"objects": [id]})).await;
    let report = &r["reports"][0];
    assert_eq!(report["printable"], json!(true), "{report}");
    assert_eq!(report["topology"]["shells"], json!(1), "{report}");

    let solid_volume = report["volume_mm3"].as_f64().unwrap();
    call(&client, "hollow", json!({"objects": [id], "wall_mm": 2})).await;
    let r = call(&client, "check_printable", json!({"objects": [id]})).await;
    let report = &r["reports"][0];
    assert_eq!(report["printable"], json!(true), "{report}");
    assert_eq!(report["topology"]["cavities"], json!(1), "{report}");
    assert!(
        report["volume_mm3"].as_f64().unwrap() < 0.6 * solid_volume,
        "{report}"
    );

    // …and out as a print file, with the raw generation kept in the library.
    let print = dir.path().join("stone.3mf");
    call(&client, "export", json!({"path": print})).await;
    assert!(std::fs::metadata(&print).unwrap().len() > 1000);
    call(
        &client,
        "save_asset",
        json!({"name": r["reports"][0]["name"], "note": "hollowed"}),
    )
    .await;
    let r = call(&client, "list_assets", json!({})).await;
    let versions = r["assets"][0]["versions"].as_array().expect("one asset");
    assert_eq!(versions.len(), 2, "{r}");
    assert_eq!(versions[0]["source"]["kind"], json!("generated"));
    assert_eq!(versions[1]["note"], json!("hollowed"));

    client.cancel().await.unwrap();
    let _ = server.await;
}
