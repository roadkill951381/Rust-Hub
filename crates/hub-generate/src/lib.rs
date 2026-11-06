//! Client for `hub-gen`, the local generation service in `services/hub-gen`.
//!
//! [`Client`] implements [`hub_core::Generator`]: it submits a job over HTTP,
//! polls its progress, and downloads the finished GLB into a cache folder.
//! The service runs on the person's own GPU, on this machine or another one
//! on their network, so nothing is billed per model.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use base64::Engine as _;
use hub_core::engine::{GenerateRequest, Generated, GenerationInfo, Generator, Progress};
use serde::{Deserialize, Serialize};

/// Where the service listens unless `HUB_GEN_URL` says otherwise.
pub const DEFAULT_URL: &str = "http://127.0.0.1:8765";

/// Largest file the client downloads (a textured GLB is typically 10 to 60 MB).
const MAX_DOWNLOAD: u64 = 1 << 30;

/// A model the service knows, from `/health`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub kind: String,
    pub available: bool,
    pub loaded: bool,
    pub vram_gb: f32,
    pub licence: String,
    pub note: String,
}

/// The service's `/health` answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Health {
    pub status: String,
    pub version: String,
    pub device: String,
    pub default_model: String,
    pub image_model: String,
    pub texture_model: String,
    pub queued: u32,
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct JobRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_b64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    texture: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    faces: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
struct JobView {
    id: String,
    status: String,
    progress: f32,
    stage: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    result: Option<JobResult>,
}

#[derive(Debug, Clone, Deserialize)]
struct JobResult {
    model: String,
    #[serde(default)]
    prompt: Option<String>,
    seed: u64,
    seconds: f32,
    textured: bool,
    files: std::collections::HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    detail: serde_json::Value,
}

/// A connection to a `hub-gen` service.
#[derive(Debug, Clone)]
pub struct Client {
    base: String,
    token: Option<String>,
    agent: ureq::Agent,
    cache: PathBuf,
    poll: Duration,
    timeout: Duration,
}

impl Client {
    /// A client for the service at `base_url`, e.g. `http://127.0.0.1:8765`.
    pub fn new(base_url: impl Into<String>) -> Self {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(120)))
            .build()
            .into();
        Self {
            base: base_url.into().trim_end_matches('/').to_owned(),
            token: None,
            agent,
            cache: std::env::temp_dir().join("rust-hub").join("generated"),
            poll: Duration::from_millis(500),
            // Long enough for a first run that downloads the model weights.
            timeout: Duration::from_secs(2 * 60 * 60),
        }
    }

    /// `HUB_GEN_URL` (default [`DEFAULT_URL`]) and `HUB_GEN_TOKEN`.
    pub fn from_env() -> Self {
        let url = std::env::var("HUB_GEN_URL").unwrap_or_else(|_| DEFAULT_URL.to_owned());
        let mut client = Self::new(url);
        client.token = std::env::var("HUB_GEN_TOKEN")
            .ok()
            .filter(|t| !t.is_empty());
        client
    }

    #[must_use]
    pub fn with_token(mut self, token: Option<String>) -> Self {
        self.token = token;
        self
    }

    /// Folder where downloaded models are kept (default: a `rust-hub` temp folder).
    #[must_use]
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache = dir.into();
        self
    }

    #[must_use]
    pub fn with_poll_interval(mut self, poll: Duration) -> Self {
        self.poll = poll;
        self
    }

    pub fn url(&self) -> &str {
        &self.base
    }

    /// Checks the service is up and lists its models.
    pub fn health(&self) -> Result<Health, String> {
        self.get_json("/health")
    }

    fn authorize<B>(&self, req: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
        match &self.token {
            Some(t) => req.header("Authorization", format!("Bearer {t}")),
            None => req,
        }
    }

    fn unreachable(&self, e: &ureq::Error) -> String {
        format!(
            "could not reach the generation service at {} ({e}). Is `hub-gen serve` running? \
             Set HUB_GEN_URL if it runs elsewhere",
            self.base
        )
    }

    fn check(mut response: ureq::http::Response<ureq::Body>) -> Result<ureq::Body, String> {
        let status = response.status();
        if status.is_success() {
            return Ok(response.into_body());
        }
        let detail = response
            .body_mut()
            .read_json::<ErrorBody>()
            .map(|b| match b.detail {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            })
            .unwrap_or_default();
        Err(match status.as_u16() {
            401 => "the generation service needs a token: set HUB_GEN_TOKEN".to_owned(),
            _ => format!("the generation service refused the request ({status}): {detail}"),
        })
    }

    fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, String> {
        let response = self
            .authorize(self.agent.get(format!("{}{path}", self.base)))
            .call()
            .map_err(|e| self.unreachable(&e))?;
        Self::check(response)?
            .read_json()
            .map_err(|e| format!("unexpected answer from {path}: {e}"))
    }

    fn download(&self, path: &str, to: &Path) -> Result<(), String> {
        let response = self
            .authorize(self.agent.get(format!("{}{path}", self.base)))
            .call()
            .map_err(|e| self.unreachable(&e))?;
        let bytes = Self::check(response)?
            .with_config()
            .limit(MAX_DOWNLOAD)
            .read_to_vec()
            .map_err(|e| format!("download of {path} failed: {e}"))?;
        std::fs::write(to, bytes).map_err(|e| format!("could not write {}: {e}", to.display()))
    }

    fn submit(&self, request: &GenerateRequest) -> Result<JobView, String> {
        let image_b64 = match &request.image {
            Some(path) => {
                let bytes = std::fs::read(path)
                    .map_err(|e| format!("could not read {}: {e}", path.display()))?;
                Some(base64::engine::general_purpose::STANDARD.encode(bytes))
            }
            None => None,
        };
        let body = JobRequest {
            prompt: request.prompt.as_deref(),
            image_b64,
            model: request.model.as_deref(),
            texture: request.texture,
            seed: request.seed,
            faces: request.faces,
        };
        let response = self
            .authorize(self.agent.post(format!("{}/v1/jobs", self.base)))
            .send_json(&body)
            .map_err(|e| self.unreachable(&e))?;
        Self::check(response)?
            .read_json()
            .map_err(|e| format!("unexpected answer from /v1/jobs: {e}"))
    }

    fn cancel(&self, id: &str) {
        let _ = self
            .authorize(self.agent.delete(format!("{}/v1/jobs/{id}", self.base)))
            .call();
    }
}

impl Generator for Client {
    fn describe(&self) -> String {
        format!("hub-gen at {}", self.base)
    }

    fn generate(
        &self,
        request: &GenerateRequest,
        progress: Progress<'_>,
    ) -> Result<Generated, String> {
        let mut job = self.submit(request)?;
        tracing::info!(id = %job.id, "generation job submitted");
        let started = Instant::now();
        loop {
            progress(job.progress, &job.stage);
            match job.status.as_str() {
                "done" => break,
                "failed" => {
                    return Err(format!(
                        "generation failed: {}",
                        job.error.as_deref().unwrap_or("no reason given")
                    ));
                }
                "cancelled" => return Err("the generation job was cancelled".into()),
                _ => {}
            }
            if started.elapsed() > self.timeout {
                self.cancel(&job.id);
                return Err(format!(
                    "generation took longer than {} minutes and was cancelled",
                    self.timeout.as_secs() / 60
                ));
            }
            std::thread::sleep(self.poll);
            job = self.get_json(&format!("/v1/jobs/{}", job.id))?;
        }

        let result = job
            .result
            .ok_or("the service reported success without a result")?;
        std::fs::create_dir_all(&self.cache)
            .map_err(|e| format!("could not create {}: {e}", self.cache.display()))?;
        let mesh_url = result
            .files
            .get("mesh.glb")
            .ok_or("the service's result has no mesh.glb")?;
        let path = self.cache.join(format!("{}.glb", job.id));
        self.download(mesh_url, &path)?;
        let image = match result.files.get("reference.png") {
            Some(url) => {
                let png = self.cache.join(format!("{}.png", job.id));
                self.download(url, &png).ok().map(|()| png)
            }
            None => None,
        };
        Ok(Generated {
            path,
            info: GenerationInfo {
                model: result.model,
                prompt: result.prompt,
                seed: result.seed,
                textured: result.textured,
                seconds: result.seconds,
                image,
            },
        })
    }
}
