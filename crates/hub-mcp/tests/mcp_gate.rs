//! Drives the MCP server through a real client over an in-memory pipe:
//! the phase 1 gate, made from tool calls only.

use hub_mcp::HubServer;
use rmcp::ServiceExt as _;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use serde_json::{Value, json};

async fn call(
    client: &rmcp::service::RunningService<rmcp::RoleClient, ()>,
    name: &'static str,
    args: Value,
) -> CallToolResult {
    let arguments = args.as_object().cloned().unwrap_or_default();
    client
        .call_tool(CallToolRequestParams::new(name).with_arguments(arguments))
        .await
        .unwrap_or_else(|e| panic!("{name} transport error: {e}"))
}

fn body(result: &CallToolResult) -> Value {
    let text = &result.content[0].as_text().expect("text result").text;
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()))
}

#[tokio::test(flavor = "multi_thread")]
async fn claude_can_prepare_a_model_for_printing() {
    let (server_io, client_io) = tokio::io::duplex(1 << 20);
    let server = tokio::spawn(async move {
        HubServer::new()
            .serve(server_io)
            .await
            .expect("server starts")
            .waiting()
            .await
    });
    let client = ().serve(client_io).await.expect("client connects");

    let tools = client.list_all_tools().await.unwrap();
    let names: Vec<_> = tools.iter().map(|t| t.name.to_string()).collect();
    for expected in [
        "import",
        "export",
        "scale_to_size",
        "check_printable",
        "render_preview",
        "undo",
        "run_commands",
    ] {
        assert!(
            names.contains(&expected.to_owned()),
            "missing tool {expected}"
        );
    }

    // Make a game asset to load: a sphere written as GLB.
    let dir = tempfile::tempdir().unwrap();
    let glb = dir.path().join("asset.glb");
    let stl = dir.path().join("print/asset.stl");
    let r = call(
        &client,
        "add_primitive",
        json!({"shape": "sphere", "size_mm": 500}),
    )
    .await;
    assert_ne!(r.is_error, Some(true), "{:?}", body(&r));
    call(&client, "export", json!({"path": glb})).await;
    call(&client, "delete", json!({"objects": [0]})).await;

    // The gate: load a GLB, scale to 80 mm, check watertight, export STL.
    let r = call(&client, "import", json!({"path": glb})).await;
    let id = body(&r)["created"][0]
        .as_u64()
        .expect("import returns an id");
    let r = call(
        &client,
        "scale_to_size",
        json!({"objects": [id], "size_mm": 80}),
    )
    .await;
    let size = body(&r)["objects"][0]["size_mm"].clone();
    assert!((size[0].as_f64().unwrap() - 80.0).abs() < 1e-3, "{size}");
    call(&client, "place_on_floor", json!({})).await;
    let r = call(&client, "check_printable", json!({})).await;
    let report = &body(&r)["reports"][0];
    assert_eq!(report["watertight"], json!(true), "{report}");
    assert_eq!(report["printable"], json!(true), "{report}");
    let r = call(&client, "export", json!({"path": stl})).await;
    assert_ne!(r.is_error, Some(true), "{:?}", body(&r));
    assert!(stl.exists());

    // Errors come back as tool errors Claude can read, not transport failures.
    let r = call(
        &client,
        "import",
        json!({"path": dir.path().join("missing.glb")}),
    )
    .await;
    assert_eq!(r.is_error, Some(true));
    assert!(body(&r).as_str().unwrap().contains("missing.glb"));

    // Undo walks back the edits.
    let r = call(&client, "undo", json!({})).await;
    assert!(
        body(&r)["summary"]
            .as_str()
            .unwrap()
            .contains("Place on floor")
    );

    // Batches run in order and stop at the first failure.
    let r = call(
        &client,
        "run_commands",
        json!({"commands": [
            {"op": "add_primitive", "shape": "cube", "size_mm": 10},
            {"op": "delete", "objects": [999]},
            {"op": "add_primitive", "shape": "cube", "size_mm": 10}
        ]}),
    )
    .await;
    assert_eq!(r.is_error, Some(true));
    assert_eq!(body(&r)["outcomes"].as_array().unwrap().len(), 1);

    client.cancel().await.unwrap();
    let _ = server.await;
}
