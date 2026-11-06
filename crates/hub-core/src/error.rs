use std::path::PathBuf;

use crate::scene::ObjectId;

/// Everything that can go wrong inside the hub core.
///
/// Messages are written for the person (or Claude) who issued the command, so
/// they name the object, file or value at fault and, where possible, the fix.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no object with id {0}")]
    UnknownObject(ObjectId),

    #[error("the scene is empty")]
    EmptyScene,

    #[error("unsupported file type for {path}: expected .glb, .gltf, .stl or .3mf")]
    UnsupportedFormat { path: PathBuf },

    #[error("could not read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid glTF file {path}: {message}")]
    Gltf { path: PathBuf, message: String },

    #[error("invalid 3MF file {path}: {message}")]
    ThreeMf { path: PathBuf, message: String },

    #[error("invalid STL file {path}: {message}")]
    Stl { path: PathBuf, message: String },

    #[error("{path} contains no triangle meshes")]
    NoGeometry { path: PathBuf },

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("nothing to undo")]
    NothingToUndo,

    #[error("nothing to redo")]
    NothingToRedo,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
