//! Runs real Blender jobs. Skips (passes) when no Blender is found; set
//! HUB_BLENDER to a blender executable or a Python with `bpy` to run them.

use hub_blender::Blender;
use hub_core::analysis::topology;
use hub_core::engine::{MeshOp, MeshTools};
use hub_core::primitives;

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
    assert!(blender.describe().starts_with("Blender"));

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
    assert!((size.y - 0.06).abs() < 0.002, "Y stays up and size is kept: {size}");
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

