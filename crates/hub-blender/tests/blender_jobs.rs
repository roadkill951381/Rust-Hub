//! Runs real Blender jobs. Skips (passes) when no Blender is found; set
//! `HUB_BLENDER` to a blender executable or a Python with `bpy` to run them.

use std::path::Path;

use hub_blender::Blender;
use hub_core::analysis::topology;
use hub_core::engine::{MeshOp, MeshTools, RigRequest, Rigger};
use hub_core::primitives;
use serde_json::Value;

fn blender() -> Option<Blender> {
    match Blender::locate() {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping Blender tests: {e}");
            None
        }
    }
}

#[test]
fn remesh_decimate_and_unwrap() {
    let Some(blender) = blender() else { return };
    assert!(MeshTools::describe(&blender).starts_with("Blender"));

    // A 60 mm tall post through a 30 mm cube: two overlapping shells that
    // voxel remeshing merges into one closed surface.
    let mut part = primitives::cylinder(0.01, 0.06, 48);
    part.append(&primitives::cube(0.03));
    assert_eq!(topology(&part).shells, 2);
    let remeshed = blender
        .apply(&part, MeshOp::VoxelRemesh { voxel: 0.001 })
        .unwrap();
    let t = topology(&remeshed);
    assert!(t.is_watertight() && t.shells == 1, "{t:?}");
    let size = remeshed.bounds().unwrap().size();
    assert!(
        (size.y - 0.06).abs() < 0.002,
        "Y stays up and size is kept: {size}"
    );
    assert!((size.x - 0.03).abs() < 0.002, "{size}");

    let fine = primitives::uv_sphere(0.04, 128, 64);
    let decimated = blender
        .apply(&fine, MeshOp::Decimate { ratio: 0.1 })
        .unwrap();
    let ratio = decimated.triangle_count() as f32 / fine.triangle_count() as f32;
    assert!((0.08..0.12).contains(&ratio), "{ratio}");
    assert!(topology(&decimated).is_watertight());

    let cube = primitives::cube(0.02);
    let unwrapped = blender
        .apply(
            &cube,
            MeshOp::UvUnwrap {
                angle_deg: 66.0,
                margin: 0.02,
            },
        )
        .unwrap();
    assert_eq!(unwrapped.uvs.len(), unwrapped.positions.len());
    assert!(
        unwrapped
            .uvs
            .iter()
            .all(|uv| (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y))
    );
}

#[test]
fn bad_parameters_come_back_as_messages() {
    let Some(blender) = blender() else { return };
    let err = blender
        .apply(&primitives::cube(1.0), MeshOp::VoxelRemesh { voxel: 1e-6 })
        .unwrap_err();
    assert!(err.contains("too fine"), "{err}");
}

/// The JSON chunk of a GLB file.
fn glb_json(path: &Path) -> Value {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..4], b"glTF", "{} is a GLB", path.display());
    let len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[20..20 + len]).unwrap()
}

/// Nodes without meshes (the skeleton), by name, with their transforms.
fn skeleton(gltf: &Value) -> Vec<(String, Value, Value, Value)> {
    let mut nodes: Vec<_> = gltf["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|n| n.get("mesh").is_none())
        .map(|n| {
            (
                n["name"].as_str().unwrap_or_default().to_owned(),
                n["translation"].clone(),
                n["rotation"].clone(),
                n["scale"].clone(),
            )
        })
        .collect();
    nodes.sort_by(|a, b| a.0.cmp(&b.0));
    nodes
}

#[test]
fn a_generated_model_is_rigged_like_the_reference() {
    let Some(blender) = blender() else { return };
    let dir = tempfile::tempdir().unwrap();
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rig_fixtures.py");
    // Reference arms 47 degrees below horizontal, the model's at 32 and
    // facing backwards.
    blender
        .run_script(
            &fixtures,
            [
                dir.path().as_os_str(),
                "47".as_ref(),
                "32".as_ref(),
                "turn".as_ref(),
            ],
        )
        .unwrap();
    let reference = dir.path().join("reference.glb");
    let output = dir.path().join("rigged.glb");
    let report = blender
        .rig(&RigRequest {
            model: dir.path().join("model.glb"),
            reference: reference.clone(),
            output: output.clone(),
            triangles: None,
            materials: Some("surv_test".into()),
        })
        .unwrap();

    assert!(report.turned, "{report:?}");
    assert_eq!(report.bones, 16, "{report:?}");
    assert_eq!(report.materials, ["surv_test.body", "surv_test.clothes"]);
    assert!(
        (0.02..0.6).contains(&report.skin_fraction),
        "head and hands are skin: {report:?}"
    );
    for (arm, offset) in &report.arm_offsets_deg {
        assert!(
            (8.0..22.0).contains(&offset.abs()),
            "the {arm} arm sat about 15 degrees off: {report:?}"
        );
    }

    let before = glb_json(&reference);
    let after = glb_json(&output);
    assert_eq!(skeleton(&before), skeleton(&after), "the skeleton is kept");
    assert_eq!(
        before["animations"][0]["channels"].as_array().map(Vec::len),
        after["animations"][0]["channels"].as_array().map(Vec::len),
        "the animation is kept"
    );
    let primitives = after["meshes"][0]["primitives"].as_array().unwrap();
    assert_eq!(primitives.len(), 2, "skin and clothes");
    assert!(primitives[0]["attributes"].get("JOINTS_0").is_some());
    // Smooth-shaded: triangles share their corners instead of each having
    // its own (a flat-shaded copy of this model has about 1,600).
    let corners =
        &after["accessors"][primitives[0]["attributes"]["POSITION"].as_u64().unwrap() as usize];
    assert!(
        corners["count"].as_u64().unwrap() < u64::from(u32::try_from(report.triangles).unwrap()),
        "{corners}"
    );

    // The reference as a pose guide: a PNG turnaround at the sheet's size.
    let sheet = dir.path().join("pose.png");
    blender.pose_views(&reference, true, false, &sheet).unwrap();
    let front = dir.path().join("front.png");
    blender.pose_views(&reference, false, true, &front).unwrap();
    assert!(front.is_file());
    let png = std::fs::read(&sheet).unwrap();
    assert_eq!(&png[1..4], b"PNG");
    let size = |at: usize| u32::from_be_bytes(png[at..at + 4].try_into().unwrap());
    assert_eq!((size(16), size(20)), (1536, 768));

    // A sharper face: the head close up, a "redrawn" picture with a blue
    // patch in the middle of the face, painted back onto the texture.
    let view = dir.path().join("face-view.png");
    blender.face_view(&output, &view).unwrap();
    let mut picture = image::open(&view).unwrap().to_rgb8();
    assert_eq!(picture.dimensions(), (1024, 1024));
    let skin = *picture.get_pixel(512, 560);
    assert!(
        skin[0] > skin[2] + 40,
        "the middle of the face is skin: {skin:?}"
    );
    for y in 400..520 {
        for x in 450..574 {
            picture.put_pixel(x, y, image::Rgb([30, 60, 200]));
        }
    }
    let redrawn = dir.path().join("face-redrawn.png");
    picture.save(&redrawn).unwrap();
    let painted = dir.path().join("rigged-face.glb");
    let texels = blender.project_face(&output, &redrawn, &painted).unwrap();
    assert!(texels > 100, "painted {texels} texels");
    assert_eq!(
        skeleton(&after),
        skeleton(&glb_json(&painted)),
        "painting keeps the skeleton"
    );
    let again = dir.path().join("face-again.png");
    blender.face_view(&painted, &again).unwrap();
    let again = image::open(&again).unwrap().to_rgb8();
    let patch = *again.get_pixel(512, 460);
    assert!(
        patch[2] > patch[0] + 80,
        "the patch is on the face now: {patch:?}"
    );
    let below = *again.get_pixel(512, 600);
    assert!(
        below[0] > below[2] + 40,
        "the rest of the face is still skin: {below:?}"
    );
}
