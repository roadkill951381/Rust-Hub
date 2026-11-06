//! Runs against a real hub-gen service when `HUB_GEN_TEST_URL` is set, e.g.
//! `hub-gen serve --demo` (no GPU needed). Skips otherwise.

use hub_core::analysis::topology;
use hub_core::engine::{GenerateRequest, Generator};
use hub_generate::Client;

fn client() -> Option<Client> {
    let Ok(url) = std::env::var("HUB_GEN_TEST_URL") else {
        eprintln!("skipping: set HUB_GEN_TEST_URL to a running hub-gen service");
        return None;
    };
    let token = std::env::var("HUB_GEN_TEST_TOKEN").ok();
    Some(Client::new(url).with_token(token))
}

#[test]
fn prompt_to_glb_through_the_service() {
    let Some(client) = client() else { return };
    let dir = tempfile::tempdir().unwrap();
    let client = client
        .with_cache_dir(dir.path())
        .with_poll_interval(std::time::Duration::from_millis(50));
    let health = client.health().unwrap();
    assert_eq!(health.status, "ok");
    assert!(health.models.iter().any(|m| m.name == "demo"));

    let stages = std::sync::Mutex::new(Vec::new());
    let request = GenerateRequest {
        prompt: Some("a tall vase".into()),
        model: Some("demo".into()),
        seed: Some(11),
        ..GenerateRequest::default()
    };
    let generated = client
        .generate(&request, &|_, stage| {
            stages.lock().unwrap().push(stage.to_owned());
        })
        .unwrap();
    assert_eq!(generated.info.seed, 11);
    assert_eq!(generated.info.model, "demo");
    assert!(generated.info.image.as_ref().is_some_and(|p| p.is_file()));
    assert!(stages.lock().unwrap().iter().any(|s| s == "done"));

    let objects = hub_core::io::import(&generated.path).unwrap();
    let mesh = &objects[0].mesh;
    assert!(topology(mesh).is_watertight(), "{:?}", topology(mesh));

    let err = client
        .generate(
            &GenerateRequest {
                prompt: Some("x".into()),
                model: Some("no-such-model".into()),
                ..GenerateRequest::default()
            },
            &|_, _| {},
        )
        .unwrap_err();
    assert!(err.contains("unknown shape model"), "{err}");
}

#[test]
fn unreachable_service_says_how_to_start_it() {
    let client = Client::new("http://127.0.0.1:9");
    let err = client.health().unwrap_err();
    assert!(err.contains("hub-gen serve"), "{err}");
}
