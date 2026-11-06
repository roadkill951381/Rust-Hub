//! `hub-app`: the Rust-Hub desktop editor.
//!
//! Usage: `hub-app [model files…]`

mod app;
mod viewport;
mod work;

fn main() -> eframe::Result {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("HUB_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let files: Vec<std::path::PathBuf> = std::env::args_os().skip(1).map(Into::into).collect();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Rust-Hub")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "Rust-Hub",
        options,
        Box::new(|cc| Ok(Box::new(app::HubApp::new(cc, files)))),
    )
}
