//! The whole check against stand-ins for PyPI, Hugging Face, GitHub, hub-gen
//! and a ComfyUI folder.

use std::path::Path;

use hub_updates::{Setup, Sources, Status, check};
use serde_json::{Value, json};

fn serve(routes: Vec<(&'static str, Value)>) -> String {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let url = format!("http://{}", server.server_addr().to_ip().unwrap());
    std::thread::spawn(move || {
        for request in server.incoming_requests() {
            let path = request.url().split('?').next().unwrap_or("").to_owned();
            let response = match routes.iter().find(|(p, _)| *p == path) {
                Some((_, body)) => tiny_http::Response::from_string(body.to_string()),
                None => tiny_http::Response::from_string("{}").with_status_code(404),
            };
            let _ = request.respond(response);
        }
    });
    url
}

fn comfy_checkout(dir: &Path) {
    let app = dir.join("ComfyUI");
    std::fs::create_dir_all(app.join("comfy")).unwrap();
    std::fs::write(app.join("main.py"), "").unwrap();
    std::fs::write(app.join("comfyui_version.py"), "__version__ = \"0.42.1\"\n").unwrap();
    let node = app.join("custom_nodes").join("ComfyUI-GGUF");
    std::fs::create_dir_all(&node).unwrap();
    std::fs::write(
        node.join("pyproject.toml"),
        "[project]\nversion = \"1.1.4\"\n",
    )
    .unwrap();
    let unet = app.join("models").join("unet");
    std::fs::create_dir_all(&unet).unwrap();
    std::fs::write(unet.join("flux1-schnell-Q8_0.gguf"), b"gguf").unwrap();
}

#[test]
fn every_piece_reports_what_to_do() {
    let releases = |vs: &[&str]| {
        let map: serde_json::Map<_, _> = vs
            .iter()
            .map(|v| ((*v).to_owned(), json!([{"yanked": false}])))
            .collect();
        json!({"info": {}, "releases": map})
    };
    let url = serve(vec![
        (
            "/pypi/fastapi/json",
            releases(&["0.115.0", "0.162.0", "0.163.0", "0.164.0rc1"]),
        ),
        (
            "/pypi/transformers/json",
            releases(&["4.48.0", "4.57.6", "5.22.1"]),
        ),
        (
            "/api/models/tencent/Hunyuan3D-2/commits/main",
            json!([
                {"id": "def4567890abcdef", "title": "Update weights", "date": "2026-11-01T10:00:00Z"},
                {"id": "abc1234567890", "title": "Initial", "date": "2025-01-21T10:00:00Z"}
            ]),
        ),
        (
            "/repos/Tencent-Hunyuan/Hunyuan3D-2/compare/f8db63096c...main",
            json!({"status": "ahead", "ahead_by": 3}),
        ),
        (
            "/repos/comfyanonymous/ComfyUI/releases/latest",
            json!({"tag_name": "v0.43.0"}),
        ),
        (
            "/v1/versions",
            json!({
                "version": "0.1.0",
                "project_dir": "/src/services/hub-gen",
                "python": "/src/services/hub-gen/.venv/bin/python",
                "extras": ["hunyuan"],
                "packages": [
                    {"name": "fastapi", "installed": "0.162.0", "wanted": ">=0.115", "url": null, "extra": null, "commit": null},
                    {"name": "transformers", "installed": "4.57.6", "wanted": ">=4.48,<5", "url": null, "extra": "hunyuan", "commit": null},
                    {"name": "hy3dgen", "installed": "2.0.2", "wanted": "",
                     "url": "git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2@f8db63096c", "extra": "hunyuan", "commit": "f8db63096c"}
                ],
                "weights": [
                    {"repo": "tencent/Hunyuan3D-2", "models": ["hunyuan3d-2", "hunyuan3d-paint"],
                     "revision": "abc1234567890", "downloaded": ["hunyuan3d-2"], "copy": null},
                    {"repo": "tencent/Hunyuan3D-2mini", "models": ["hunyuan3d-2mini"],
                     "revision": null, "downloaded": [], "copy": null}
                ]
            }),
        ),
    ]);
    let comfy = tempfile::tempdir().unwrap();
    comfy_checkout(comfy.path());
    let setup = Setup {
        source: None,
        built_commit: None,
        hub_gen: hub_generate::Client::new(url.clone()),
        hub_gen_dir: None,
        comfy: hub_comfy::Config {
            url: "http://127.0.0.1:9".into(),
            dir: Some(comfy.path().to_owned()),
            ..hub_comfy::Config::default()
        },
        blender: false,
        tools: false,
        sources: Sources {
            pypi: url.clone(),
            huggingface: url.clone(),
            github: url.clone(),
            blender: url,
        },
    };
    let report = check(&setup);
    let item = |id: &str| {
        report
            .find(id)
            .unwrap_or_else(|| panic!("no {id} in {report:#?}"))
    };

    let fastapi = item("hub-gen/fastapi");
    assert_eq!(fastapi.status, Status::Available);
    assert_eq!(
        fastapi.latest.as_deref(),
        Some("0.163.0"),
        "pre-releases are skipped"
    );
    let step = &fastapi.upgrade.as_ref().unwrap().steps[0];
    assert_eq!(step.program, "uv");
    assert_eq!(
        step.args,
        ["sync", "--extra", "hunyuan", "--upgrade-package", "fastapi"]
    );
    assert!(fastapi.blocked.is_some(), "hub-gen is running");
    assert!(!fastapi.can_update());

    let transformers = item("hub-gen/transformers");
    assert_eq!(transformers.status, Status::HeldBack);
    assert_eq!(transformers.latest.as_deref(), Some("5.22.1"));

    let hy3dgen = item("hub-gen/hy3dgen");
    assert_eq!(hy3dgen.status, Status::HeldBack);
    assert!(
        hy3dgen
            .latest
            .as_deref()
            .unwrap()
            .starts_with("3 newer commits")
    );

    let weights = item("weights/tencent/hunyuan3d-2");
    assert_eq!(weights.status, Status::Available);
    assert!(
        weights
            .detail
            .as_deref()
            .unwrap()
            .contains("\"Update weights\"")
    );
    assert_eq!(
        weights.upgrade.as_ref().unwrap().steps[0].args,
        ["run", "--no-sync", "hub-gen", "pull", "tencent/Hunyuan3D-2"]
    );
    assert!(weights.can_update());
    assert_eq!(
        item("weights/tencent/hunyuan3d-2mini").status,
        Status::NotInstalled
    );

    let core = item("comfyui");
    assert_eq!(core.status, Status::Available);
    assert_eq!(core.installed.as_deref(), Some("0.42.1"));
    assert_eq!(core.latest.as_deref(), Some("0.43.0"));
    let steps = &core.upgrade.as_ref().unwrap().steps;
    assert_eq!(steps[0].args, ["pull", "--ff-only"]);
    assert!(core.can_update(), "ComfyUI is not running");

    let gguf = item("comfyui/comfyui-gguf");
    assert_eq!(gguf.status, Status::Unknown);
    assert_eq!(gguf.installed.as_deref(), Some("1.1.4"));

    assert_eq!(
        item("comfyui-model/flux1-schnell-q8_0.gguf").status,
        Status::Current
    );
    let hunyuan = item("comfyui-model/hunyuan_3d_v2.1.safetensors");
    assert_eq!(hunyuan.status, Status::NotInstalled);
    let curl = &hunyuan.upgrade.as_ref().unwrap().steps[0];
    assert_eq!(curl.program, "curl");
    assert!(
        curl.args.last().unwrap().ends_with(
            "Comfy-Org/hunyuan3D_2.1_repackaged/resolve/main/hunyuan_3d_v2.1.safetensors"
        )
    );
    let ae = item("comfyui-model/ae.safetensors");
    assert!(
        ae.upgrade.as_ref().unwrap().steps.is_empty(),
        "gated: a link, not a download"
    );

    assert_eq!(item("rust-hub").status, Status::Unknown);
    let summary = report.summary();
    assert!(
        summary.starts_with("3 updates available, 2 held back"),
        "{summary}"
    );

    // The report is what the MCP tool and `hub updates --json` return.
    let text = serde_json::to_string(&report).unwrap();
    let back: hub_updates::Report = serde_json::from_str(&text).unwrap();
    assert_eq!(back, report);
}
