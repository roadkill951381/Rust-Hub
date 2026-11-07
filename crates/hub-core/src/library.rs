//! The asset library: saved models with every version kept.
//!
//! On disk it is plain files a person can browse:
//!
//! ```text
//! <root>/
//!   knight/
//!     asset.json      name, and one record per version (how it was made, size, note)
//!     v0001.glb
//!     v0002.glb
//! ```
//!
//! The default root is `~/RustHub/library`, or `HUB_LIBRARY` when set.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::engine::GenerationInfo;
use crate::error::{Error, Result};

const INDEX: &str = "asset.json";

/// How a version came to be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssetSource {
    /// Saved from the scene.
    Saved,
    /// Made by the generation service.
    Generated(Box<GenerationInfo>),
}

/// One saved version of an asset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AssetVersion {
    pub version: u32,
    /// File name inside the asset's folder.
    pub file: String,
    /// Seconds since 1970-01-01 UTC.
    pub created_unix: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub source: AssetSource,
    pub triangles: usize,
    /// Width, depth and height in mm.
    pub size_mm: [f32; 3],
}

/// An asset and all of its versions, oldest first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Asset {
    pub name: String,
    /// Folder name, derived from the name.
    pub slug: String,
    pub versions: Vec<AssetVersion>,
}

impl Asset {
    pub fn latest(&self) -> Option<&AssetVersion> {
        self.versions.last()
    }
}

/// What to record about a new version; the library fills in the number,
/// file name and time.
#[derive(Debug, Clone, PartialEq)]
pub struct NewVersion {
    pub note: Option<String>,
    pub source: AssetSource,
    pub triangles: usize,
    pub size_mm: [f32; 3],
}

/// A folder of assets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Library {
    root: PathBuf,
}

impl Library {
    /// A library at `root`. Nothing is created until the first save.
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `HUB_LIBRARY`, else `~/RustHub/library`.
    pub fn default_root() -> Option<PathBuf> {
        if let Some(dir) = std::env::var_os("HUB_LIBRARY") {
            return Some(PathBuf::from(dir));
        }
        std::env::home_dir().map(|home| home.join("RustHub").join("library"))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Saves a new version of `name`. `write` must write the model to the
    /// path it is given (a `.glb`).
    pub fn save(
        &self,
        name: &str,
        new: NewVersion,
        write: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<(Asset, PathBuf)> {
        let slug = slug(name)?;
        let dir = self.root.join(&slug);
        std::fs::create_dir_all(&dir).map_err(|source| Error::Write {
            path: dir.clone(),
            source,
        })?;
        let mut asset = self.read(&slug)?.unwrap_or_else(|| Asset {
            name: name.trim().to_owned(),
            slug: slug.clone(),
            versions: Vec::new(),
        });
        let version = asset.latest().map_or(1, |v| v.version + 1);
        let file = format!("v{version:04}.glb");
        let path = dir.join(&file);
        write(&path)?;
        asset.versions.push(AssetVersion {
            version,
            file,
            created_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            note: new.note,
            source: new.source,
            triangles: new.triangles,
            size_mm: new.size_mm,
        });
        self.write_index(&asset)?;
        Ok((asset, path))
    }

    /// Every asset whose name contains `query` (case-insensitive), by name.
    pub fn list(&self, query: Option<&str>) -> Result<Vec<Asset>> {
        let entries = match std::fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => {
                return Err(Error::Read {
                    path: self.root.clone(),
                    source,
                });
            }
        };
        let needle = query.map(str::to_lowercase);
        let mut assets = Vec::new();
        for entry in entries.flatten() {
            let Some(slug) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if let Some(asset) = self.read(&slug)?
                && needle
                    .as_deref()
                    .is_none_or(|n| asset.name.to_lowercase().contains(n) || asset.slug.contains(n))
            {
                assets.push(asset);
            }
        }
        assets.sort_by_cached_key(|a| a.name.to_lowercase());
        Ok(assets)
    }

    /// The asset called `name` and the file of `version` (the latest when `None`).
    pub fn find(&self, name: &str, version: Option<u32>) -> Result<(Asset, AssetVersion, PathBuf)> {
        let slug = slug(name)?;
        let asset = self.read(&slug)?.ok_or_else(|| {
            Error::Library(format!(
                "no asset called {name:?} in {}",
                self.root.display()
            ))
        })?;
        let chosen = match version {
            Some(n) => asset
                .versions
                .iter()
                .find(|v| v.version == n)
                .ok_or_else(|| {
                    Error::Library(format!(
                        "{} has versions 1 to {}, not {n}",
                        asset.name,
                        asset.versions.len()
                    ))
                })?,
            None => asset
                .latest()
                .ok_or_else(|| Error::Library(format!("{} has no versions", asset.name)))?,
        }
        .clone();
        let path = self.root.join(&asset.slug).join(&chosen.file);
        Ok((asset, chosen, path))
    }

    fn read(&self, slug: &str) -> Result<Option<Asset>> {
        let path = self.root.join(slug).join(INDEX);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(Error::Read { path, source }),
        };
        serde_json::from_str(&text).map(Some).map_err(|e| {
            Error::Library(format!(
                "{} is not a valid asset index: {e}",
                path.display()
            ))
        })
    }

    fn write_index(&self, asset: &Asset) -> Result<()> {
        let dir = self.root.join(&asset.slug);
        let path = dir.join(INDEX);
        let tmp = dir.join(format!("{INDEX}.tmp"));
        let json = serde_json::to_string_pretty(asset).expect("assets serialize");
        std::fs::write(&tmp, json)
            .and_then(|()| std::fs::rename(&tmp, &path))
            .map_err(|source| Error::Write { path, source })
    }
}

/// Folder name for an asset: lowercase letters, digits and dashes.
fn slug(name: &str) -> Result<String> {
    let mut out = String::new();
    for c in name.trim().chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-').to_owned();
    if out.is_empty() {
        Err(Error::InvalidArgument(format!(
            "{name:?} is not a usable asset name; use letters or digits"
        )))
    } else {
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version() -> NewVersion {
        NewVersion {
            note: None,
            source: AssetSource::Saved,
            triangles: 12,
            size_mm: [1.0, 2.0, 3.0],
        }
    }

    #[test]
    fn versions_accumulate_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let lib = Library::open(dir.path().join("lib"));
        assert_eq!(lib.list(None).unwrap(), Vec::<Asset>::new());
        for _ in 0..2 {
            lib.save("Big Knight!", version(), |p| {
                std::fs::write(p, b"glb").map_err(|source| Error::Write {
                    path: p.into(),
                    source,
                })
            })
            .unwrap();
        }
        let assets = lib.list(Some("knight")).unwrap();
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0].slug, "big-knight");
        assert_eq!(assets[0].versions.len(), 2);
        let (_, v, path) = lib.find("big knight", None).unwrap();
        assert_eq!(v.version, 2);
        assert!(path.ends_with("big-knight/v0002.glb"));
        assert!(lib.find("big knight", Some(9)).is_err());
        assert!(lib.find("dragon", None).is_err());
        assert_eq!(lib.list(Some("dragon")).unwrap(), Vec::<Asset>::new());
        assert!(slug("!!").is_err());
    }
}
