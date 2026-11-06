//! Rust-Hub core: the scene model, the command layer with undo, file
//! import/export and 3D-print checks. No UI and no GPU code lives here, so the
//! desktop app, the CLI and the MCP server all share exactly this behaviour.
//!
//! ```
//! use hub_core::{Command, Session, primitives::Primitive};
//!
//! let mut session = Session::new();
//! session.execute(Command::AddPrimitive { shape: Primitive::Sphere, size_mm: 30.0, name: None })?;
//! session.execute(Command::ScaleToSize { objects: None, size_mm: 80.0, axis: Default::default() })?;
//! let report = session.execute(Command::CheckPrintable { objects: None, min_wall_mm: None })?;
//! assert!(report.reports[0].printable);
//! # Ok::<(), hub_core::Error>(())
//! ```

pub mod analysis;
pub mod bvh;
pub mod command;
pub mod engine;
pub mod error;
pub mod hollow;
pub mod io;
pub mod library;
pub mod mesh;
pub mod primitives;
pub mod scene;
pub mod session;

pub use command::{Command, ObjectInfo, Outcome, SavedAsset, SizeAxis};
pub use engine::{
    Engines, GenerateRequest, Generated, GenerationInfo, Generator, MeshOp, MeshTools,
};
pub use error::{Error, Result};
pub use mesh::{Aabb, Mesh};
pub use scene::{Material, Object, ObjectId, Scene, Transform};
pub use session::{Job, JobOutput, Session, Step};
