//! The public places newer versions are announced: PyPI, Hugging Face,
//! GitHub and Blender's download site.

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;

use crate::Version;

/// Base addresses, overridable so tests can stand in for each site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sources {
    pub pypi: String,
    pub huggingface: String,
    pub github: String,
    pub blender: String,
}

impl Default for Sources {
    fn default() -> Self {
        Self {
            pypi: "https://pypi.org".into(),
            huggingface: "https://huggingface.co".into(),
            github: "https://api.github.com".into(),
            blender: "https://projects.blender.org".into(),
        }
    }
}

/// A commit on a Hugging Face repo.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct HfCommit {
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub date: String,
}

/// How a GitHub branch compares with a commit.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Compare {
    /// `ahead`, `behind`, `identical` or `diverged`.
    pub status: String,
    pub ahead_by: u64,
}

#[derive(Debug)]
pub struct Web {
    sources: Sources,
    agent: ureq::Agent,
    github_token: Option<String>,
    hf_token: Option<String>,
}

impl Web {
    pub fn new(sources: Sources) -> Self {
        let tls = ureq::tls::TlsConfig::builder()
            // The system's certificates, so it works behind antivirus or a
            // company proxy that the system trusts.
            .root_certs(ureq::tls::RootCerts::PlatformVerifier)
            .build();
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(30)))
            .user_agent(concat!("rust-hub/", env!("CARGO_PKG_VERSION")))
            .tls_config(tls)
            .build()
            .into();
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        Self {
            sources,
            agent,
            github_token: var("HUB_GITHUB_TOKEN").or_else(|| var("GITHUB_TOKEN")),
            hf_token: var("HF_TOKEN").or_else(|| var("HUGGING_FACE_HUB_TOKEN")),
        }
    }

    fn get(&self, url: &str, token: Option<&str>, accept: &str) -> Result<ureq::Body, String> {
        let mut req = self.agent.get(url).header("Accept", accept);
        if let Some(t) = token {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        let response = req
            .call()
            .map_err(|e| format!("could not reach {url}: {e}"))?;
        let status = response.status();
        if status.is_success() {
            Ok(response.into_body())
        } else {
            Err(match status.as_u16() {
                401 | 403 if url.starts_with(&self.sources.github) => format!(
                    "GitHub refused {url} ({status}); its hourly limit for anonymous checks \
                     may be used up: wait, or set HUB_GITHUB_TOKEN"
                ),
                401 | 403 if url.starts_with(&self.sources.huggingface) => {
                    format!("{url} needs a Hugging Face sign-in ({status}); set HF_TOKEN")
                }
                _ => format!("{url} answered {status}"),
            })
        }
    }

    fn json<T: serde::de::DeserializeOwned>(
        &self,
        url: &str,
        token: Option<&str>,
    ) -> Result<T, String> {
        self.get(url, token, "application/json")?
            .with_config()
            .limit(16 << 20)
            .read_json()
            .map_err(|e| format!("unexpected answer from {url}: {e}"))
    }

    /// All release numbers of a PyPI package, leaving out fully yanked ones.
    pub fn pypi_versions(&self, package: &str) -> Result<Vec<String>, String> {
        #[derive(Deserialize)]
        struct File {
            #[serde(default)]
            yanked: bool,
        }
        #[derive(Deserialize)]
        struct Project {
            releases: HashMap<String, Vec<File>>,
        }
        let url = format!("{}/pypi/{package}/json", self.sources.pypi);
        let project: Project = self.json(&url, None)?;
        Ok(project
            .releases
            .into_iter()
            .filter(|(_, files)| !files.is_empty() && !files.iter().all(|f| f.yanked))
            .map(|(v, _)| v)
            .collect())
    }

    /// The newest commits on a Hugging Face model repo's main branch, newest
    /// first (one page, about 50).
    pub fn hf_commits(&self, repo: &str) -> Result<Vec<HfCommit>, String> {
        let url = format!(
            "{}/api/models/{repo}/commits/main",
            self.sources.huggingface
        );
        self.json(&url, self.hf_token.as_deref())
    }

    /// The tag of a GitHub repo's latest release.
    pub fn github_release(&self, repo: &str) -> Result<String, String> {
        #[derive(Deserialize)]
        struct Release {
            tag_name: String,
        }
        let url = format!("{}/repos/{repo}/releases/latest", self.sources.github);
        Ok(self
            .json::<Release>(&url, self.github_token.as_deref())?
            .tag_name)
    }

    /// How far `branch` has moved on from `commit`.
    pub fn github_compare(
        &self,
        repo: &str,
        commit: &str,
        branch: &str,
    ) -> Result<Compare, String> {
        let url = format!(
            "{}/repos/{repo}/compare/{commit}...{branch}",
            self.sources.github
        );
        self.json(&url, self.github_token.as_deref())
    }

    /// The newest Blender release, e.g. `5.2.3`, from the release tags on
    /// projects.blender.org (download.blender.org turns scripts away).
    pub fn blender_latest(&self) -> Result<String, String> {
        #[derive(Deserialize)]
        struct Tag {
            name: String,
        }
        let url = format!(
            "{}/api/v1/repos/blender/blender/tags?limit=50",
            self.sources.blender
        );
        // Newest first by date, mixing the current and LTS series.
        let tags: Vec<Tag> = self.json(&url, None)?;
        tags.iter()
            .filter_map(|t| {
                let v = Version::parse(&t.name)?;
                (!v.is_prerelease()).then(|| (v, t.name.trim_start_matches('v').to_owned()))
            })
            .max_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, name)| name)
            .ok_or_else(|| "Blender's tag list has no release versions".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blender_tags_are_sorted_by_version() {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let url = format!("http://{}", server.server_addr().to_ip().unwrap());
        std::thread::spawn(move || {
            for request in server.incoming_requests() {
                let body = r#"[{"name":"v4.5.15"},{"name":"v5.2.2"},{"name":"v5.3.0-rc1"},
                    {"name":"v5.2.3"},{"name":"v4.2.23"}]"#;
                let _ = request.respond(tiny_http::Response::from_string(body));
            }
        });
        let web = Web::new(Sources {
            blender: url,
            ..Sources::default()
        });
        assert_eq!(web.blender_latest().unwrap(), "5.2.3");
    }
}
