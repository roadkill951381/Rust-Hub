//! Every format round-trips geometry, scale and orientation.

use hub_core::analysis::check_printable;
use hub_core::primitives::Primitive;
use hub_core::{Command, ObjectId, Session};

/// A 20 × 60 × 20 mm cylinder standing on its end, moved off the origin.
fn tall_cylinder() -> Session {
    let mut s = Session::new();
    s.execute(Command::AddPrimitive {
        shape: Primitive::Cylinder,
        size_mm: 20.0,
        name: Some("tower".into()),
    })
    .unwrap();
    s.execute(Command::SetTransform {
        object: ObjectId(0),
        position_mm: Some([5.0, 30.0, -7.0]),
        rotation_deg: None,
        scale: Some([1.0, 3.0, 1.0]),
    })
    .unwrap();
    s
}

fn round_trip(ext: &str) -> (Session, Session) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("tower.{ext}"));
    let mut original = tall_cylinder();
    original
        .execute(Command::Export {
            path: path.clone(),
            objects: None,
        })
        .unwrap();
    let mut loaded = Session::new();
    loaded.execute(Command::Import { path }).unwrap();
    (original, loaded)
}

#[test]
fn every_format_keeps_size_and_up_axis() {
    for ext in ["glb", "gltf", "stl", "3mf"] {
        let (original, loaded) = round_trip(ext);
        let a = original.scene().bounds().unwrap();
        let b = loaded.scene().bounds().unwrap();
        assert!(
            a.min.abs_diff_eq(b.min, 1e-5) && a.max.abs_diff_eq(b.max, 1e-5),
            "{ext}: bounds {a:?} vs {b:?}"
        );
        // Height stays on the up axis (scene Y).
        assert!(
            (b.size().y - 0.06).abs() < 1e-5,
            "{ext}: height {}",
            b.size().y
        );
    }
}

#[test]
fn print_formats_stay_watertight() {
    for ext in ["stl", "3mf"] {
        let (_, loaded) = round_trip(ext);
        let report = check_printable(&loaded.scene().objects()[0]);
        assert!(report.printable, "{ext}: {:?}", report.issues);
        assert!(
            (report.size_mm[2] - 60.0).abs() < 1e-2,
            "{ext}: {:?}",
            report.size_mm
        );
    }
}

#[test]
fn gltf_keeps_transforms_names_and_materials() {
    let (original, loaded) = round_trip("glb");
    let a = &original.scene().objects()[0];
    let b = &loaded.scene().objects()[0];
    assert_eq!(b.name, "tower");
    assert!(
        a.transform
            .translation
            .abs_diff_eq(b.transform.translation, 1e-6)
    );
    assert!(a.transform.scale.abs_diff_eq(b.transform.scale, 1e-6));
    assert_eq!(a.material, b.material);
    assert_eq!(a.mesh.indices, b.mesh.indices);
}

#[test]
fn stl_is_written_in_millimetres_z_up() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tower.stl");
    tall_cylinder()
        .execute(Command::Export {
            path: path.clone(),
            objects: None,
        })
        .unwrap();
    let stl = stl_io::read_stl(&mut std::fs::File::open(&path).unwrap()).unwrap();
    let zs = stl.vertices.iter().map(|v| v.0[2]);
    let (lo, hi) = zs.fold((f32::MAX, f32::MIN), |(lo, hi), z| (lo.min(z), hi.max(z)));
    assert!((hi - lo - 60.0).abs() < 1e-3, "z extent {}", hi - lo);
}

#[test]
fn import_errors_name_the_file() {
    let mut s = Session::new();
    let err = s
        .execute(Command::Import {
            path: "does/not/exist.glb".into(),
        })
        .unwrap_err()
        .to_string();
    assert!(err.contains("does/not/exist.glb"), "{err}");
    let err = s
        .execute(Command::Import {
            path: "model.obj".into(),
        })
        .unwrap_err()
        .to_string();
    assert!(err.contains(".glb"), "{err}");
}
