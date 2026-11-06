//! Renders a frame off-screen. Skips (passes) on machines with no GPU adapter
//! at all; CI installs a software Vulkan driver so it runs there.

use hub_core::primitives::Primitive;
use hub_core::{Command, Session};
use hub_render::{FrameOptions, Headless, framing_camera};

#[test]
fn renders_a_png_with_the_object_in_view() {
    let mut renderer = match Headless::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let mut session = Session::new();
    session
        .execute(Command::AddPrimitive {
            shape: Primitive::Sphere,
            size_mm: 40.0,
            name: None,
        })
        .unwrap();
    let scene = session.scene();
    let png = renderer
        .render_png(
            scene,
            &framing_camera(scene),
            160,
            120,
            FrameOptions::default(),
        )
        .unwrap();
    let image = image::load_from_memory(&png).unwrap().to_rgba8();
    assert_eq!(image.dimensions(), (160, 120));
    // The centre pixel is the lit sphere, much brighter than the background corner.
    let center = image.get_pixel(80, 60)[0];
    let corner = image.get_pixel(2, 2)[0];
    assert!(center > corner + 60, "center {center}, corner {corner}");
}
