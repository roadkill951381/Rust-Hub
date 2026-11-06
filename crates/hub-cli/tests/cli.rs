//! The `hub` binary end to end: script, convert, check.

use std::process::Command;

fn hub() -> Command {
    Command::new(env!("CARGO_BIN_EXE_hub"))
}

#[test]
fn script_convert_and_check() {
    let dir = tempfile::tempdir().unwrap();
    let glb = dir.path().join("ball.glb");
    let script = dir.path().join("make.json");
    std::fs::write(
        &script,
        serde_json::json!([
            {"op": "add_primitive", "shape": "sphere", "size_mm": 300},
            {"op": "export", "path": glb}
        ])
        .to_string(),
    )
    .unwrap();
    let out = hub().arg("run").arg(&script).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stl = dir.path().join("ball.stl");
    let out = hub()
        .args(["--json", "convert"])
        .arg(&glb)
        .arg(&stl)
        .args(["--size-mm", "80", "--place-on-floor"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = hub().args(["--json", "check"]).arg(&stl).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let r = &report["reports"][0];
    assert_eq!(r["printable"], true, "{r}");
    let height = r["size_mm"][2].as_f64().unwrap();
    assert!((height - 80.0).abs() < 0.01, "height {height}");
}

#[test]
fn check_exits_2_for_open_meshes() {
    let dir = tempfile::tempdir().unwrap();
    // A single triangle cannot enclose a solid.
    let stl = dir.path().join("open.stl");
    std::fs::write(
        &stl,
        "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 10 0 0\nvertex 0 10 0\nendloop\nendfacet\nendsolid t\n",
    )
    .unwrap();
    let out = hub().arg("check").arg(&stl).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stdout).contains("open edges"));
}

#[test]
fn bad_scripts_fail_with_the_command_index() {
    let out = hub()
        .args(["run", "-"])
        .stdin(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write as _;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(br#"[{"op":"undo"}]"#)?;
            child.wait_with_output()
        })
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("command 0 failed"));
}
