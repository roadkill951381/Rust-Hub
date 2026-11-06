//! One place that decides which engines a front end gets, so the desktop app,
//! the CLI and the MCP server behave the same.
//!
//! | Engine | Found through |
//! |---|---|
//! | Blender | `HUB_BLENDER`, the `PATH`, standard install folders (on first use) |
//! | Generation service | `HUB_GEN_URL` (default `http://127.0.0.1:8765`), `HUB_GEN_TOKEN` |
//! | Asset library | `HUB_LIBRARY` (default `~/RustHub/library`) |
//!
//! Nothing is contacted or created here; each engine reports its own setup
//! problem when a command first needs it.

use std::path::PathBuf;
use std::sync::Arc;

use hub_blender::LazyBlender;
use hub_core::Engines;
use hub_core::library::Library;
use hub_generate::Client;

/// Overrides for [`build`]; `None` fields fall back to the environment.
#[derive(Debug, Clone, Default)]
pub struct Config {
    pub library: Option<PathBuf>,
    pub generator_url: Option<String>,
}

/// Engines from the environment.
pub fn from_env() -> Engines {
    build(&Config::default())
}

pub fn build(config: &Config) -> Engines {
    let mut generator = Client::from_env();
    if let Some(url) = &config.generator_url {
        generator = Client::new(url.clone()).with_token(std::env::var("HUB_GEN_TOKEN").ok());
    }
    let library = config
        .library
        .clone()
        .or_else(Library::default_root)
        .map(Library::open);
    Engines {
        mesh_tools: Some(Arc::new(LazyBlender::new())),
        generator: Some(Arc::new(generator)),
        library,
    }
}
