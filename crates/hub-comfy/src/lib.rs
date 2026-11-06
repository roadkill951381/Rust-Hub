//! ComfyUI as a Rust-Hub engine: the person's own ComfyUI, its models and
//! their workflows make pictures and 3D models for the hub.
//!
//! [`ComfyUi`] talks to ComfyUI's HTTP API: it uploads the input picture,
//! queues a [workflow](workflows) with the prompt, seed and picture filled in,
//! waits for it, and downloads what it saved. When ComfyUI is not running and
//! its folder is known, it starts it. After a job it can ask ComfyUI to unload
//! its models, so the GPU is free for the next engine.
//!
//! It implements [`Generator`] for a whole job in ComfyUI (picture, then
//! shape), and [`ComfyUi::make_picture`] lets the hub pair ComfyUI's picture
//! with another engine's shape model.

pub mod workflows;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use hub_core::engine::{GenerateRequest, Generated, GenerationInfo, Generator, Progress};
use serde_json::{Value, json};

pub use workflows::{
    FLUX_DEV, FLUX_SCHNELL, HUNYUAN3D_21, MODEL_FILES, Makes, ModelFile, Workflow,
};

/// Where ComfyUI listens unless `HUB_COMFY_URL` says otherwise.
pub const DEFAULT_URL: &str = "http://127.0.0.1:8188";

/// Largest file downloaded from ComfyUI.
const MAX_DOWNLOAD: u64 = 1 << 30;

/// Settings for [`ComfyUi`].
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub url: String,
    /// The ComfyUI folder (portable or a git checkout), so the hub can start
    /// it when it is not running.
    pub dir: Option<PathBuf>,
    /// Folder of the person's own API-format workflows, `<name>.json`.
    pub workflows: Option<PathBuf>,
    /// Workflow that makes a picture from a prompt.
    pub picture_workflow: String,
    /// Workflow that makes a 3D model from a picture.
    pub model_workflow: String,
    /// Unload ComfyUI's models after each job so the GPU is free again.
    pub free_after: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            url: DEFAULT_URL.to_owned(),
            dir: None,
            workflows: None,
            picture_workflow: FLUX_SCHNELL.to_owned(),
            model_workflow: HUNYUAN3D_21.to_owned(),
            free_after: true,
        }
    }
}

impl Config {
    /// From `HUB_COMFY_URL`, `HUB_COMFY_DIR` (else a ComfyUI found in the usual
    /// places), `HUB_COMFY_WORKFLOWS` (else `~/RustHub/comfyui`),
    /// `HUB_COMFY_PICTURE_WORKFLOW`, `HUB_COMFY_MODEL_WORKFLOW` and
    /// `HUB_COMFY_KEEP_LOADED`.
    pub fn from_env() -> Self {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let home = std::env::home_dir();
        let defaults = Self::default();
        Self {
            url: var("HUB_COMFY_URL").unwrap_or(defaults.url),
            dir: var("HUB_COMFY_DIR")
                .map(PathBuf::from)
                .or_else(find_install),
            workflows: var("HUB_COMFY_WORKFLOWS")
                .map(PathBuf::from)
                .or_else(|| home.map(|h| h.join("RustHub").join("comfyui"))),
            picture_workflow: var("HUB_COMFY_PICTURE_WORKFLOW")
                .unwrap_or(defaults.picture_workflow),
            model_workflow: var("HUB_COMFY_MODEL_WORKFLOW").unwrap_or(defaults.model_workflow),
            free_after: var("HUB_COMFY_KEEP_LOADED")
                .is_none_or(|v| matches!(v.to_ascii_lowercase().as_str(), "0" | "false" | "no")),
        }
    }
}

/// Where ComfyUI is installed, for update checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Installation {
    /// The folder with `main.py`, `models` and `custom_nodes`.
    pub app: PathBuf,
    /// The portable build's folder (with `python_embeded` and `update`).
    pub portable: Option<PathBuf>,
    /// The Python that runs it.
    pub python: PathBuf,
}

impl Config {
    /// The ComfyUI install in [`Config::dir`], if one is there.
    pub fn installation(&self) -> Option<Installation> {
        let launch = Launch::find(self.dir.as_deref()?, 0)?;
        let portable = launch
            .args
            .iter()
            .any(|a| a == "--windows-standalone-build");
        Some(Installation {
            app: if portable {
                launch.cwd.join("ComfyUI")
            } else {
                launch.cwd.clone()
            },
            portable: portable.then_some(launch.cwd),
            python: launch.program,
        })
    }
}

/// A file ComfyUI saved, as its history reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedFile {
    pub filename: String,
    pub subfolder: String,
    pub kind: String,
}

/// What a finished workflow saved.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Saved {
    pub pictures: Vec<SavedFile>,
    pub models: Vec<SavedFile>,
}

/// A connection to a ComfyUI.
#[derive(Debug, Clone)]
pub struct ComfyUi {
    config: Config,
    base: String,
    agent: ureq::Agent,
    cache: PathBuf,
    poll: Duration,
    start_timeout: Duration,
    job_timeout: Duration,
}

impl ComfyUi {
    pub fn new(config: Config) -> Self {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(120)))
            .build()
            .into();
        Self {
            base: config.url.trim_end_matches('/').to_owned(),
            config,
            agent,
            cache: std::env::temp_dir().join("rust-hub").join("comfyui"),
            poll: Duration::from_millis(500),
            start_timeout: Duration::from_secs(5 * 60),
            job_timeout: Duration::from_secs(2 * 60 * 60),
        }
    }

    pub fn from_env() -> Self {
        Self::new(Config::from_env())
    }

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

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn url(&self) -> &str {
        &self.base
    }

    /// Whether ComfyUI answers at its URL.
    pub fn is_running(&self) -> bool {
        self.agent
            .get(format!("{}/system_stats", self.base))
            .call()
            .is_ok_and(|r| r.status().is_success())
    }

    /// Makes sure ComfyUI is up, starting it from its folder if needed.
    pub fn ensure_running(&self, progress: Progress<'_>) -> Result<(), String> {
        if self.is_running() {
            return Ok(());
        }
        let Some(dir) = &self.config.dir else {
            return Err(format!(
                "ComfyUI is not running at {}. Start it, or set HUB_COMFY_DIR to its folder \
                 so the hub can start it when needed",
                self.base
            ));
        };
        let launch = Launch::find(dir, port_of(&self.base)).ok_or_else(|| {
            format!(
                "{} does not look like a ComfyUI folder (no main.py, and no \
                 python_embeded with ComfyUI/main.py); set HUB_COMFY_DIR to the right one",
                dir.display()
            )
        })?;
        progress(0.0, "starting ComfyUI");
        tracing::info!(program = %launch.program.display(), "starting ComfyUI");
        let mut child = launch.spawn(&self.cache)?;
        let started = Instant::now();
        loop {
            if self.is_running() {
                return Ok(());
            }
            if let Ok(Some(status)) = child.try_wait() {
                return Err(format!(
                    "ComfyUI stopped while starting ({status}). Start it by hand to see why; {}",
                    launch.log_hint(&self.cache)
                ));
            }
            if started.elapsed() > self.start_timeout {
                return Err(format!(
                    "ComfyUI did not answer at {} within {} minutes of starting it",
                    self.base,
                    self.start_timeout.as_secs() / 60
                ));
            }
            std::thread::sleep(Duration::from_secs(1));
        }
    }

    /// Asks ComfyUI to unload its models and free the GPU, and waits briefly
    /// for the memory to come back.
    pub fn free_memory(&self) {
        let before = self.vram_free();
        let sent = self
            .agent
            .post(format!("{}/free", self.base))
            .send_json(json!({"unload_models": true, "free_memory": true}))
            .is_ok_and(|r| r.status().is_success());
        if !sent {
            return;
        }
        let Some(before) = before else { return };
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(15) {
            std::thread::sleep(Duration::from_millis(500));
            if self
                .vram_free()
                .is_some_and(|now| now > before + (512 << 20))
            {
                return;
            }
        }
    }

    fn vram_free(&self) -> Option<u64> {
        let mut r = self
            .agent
            .get(format!("{}/system_stats", self.base))
            .call()
            .ok()?;
        let stats: Value = r.body_mut().read_json().ok()?;
        stats["devices"][0]["vram_free"].as_u64()
    }

    /// Makes a picture of `prompt` with the picture workflow (or `workflow`)
    /// and returns where it was saved and the workflow's name.
    pub fn make_picture(
        &self,
        prompt: &str,
        seed: u64,
        workflow: Option<&str>,
        progress: Progress<'_>,
    ) -> Result<(PathBuf, String), String> {
        self.ensure_running(progress)?;
        let name = workflow.unwrap_or(&self.config.picture_workflow);
        let workflow = self.workflow(name)?;
        let saved = self.run(
            &workflow,
            Some(prompt),
            None,
            seed,
            "making the picture in ComfyUI",
            progress,
        )?;
        let file = saved.pictures.last().ok_or_else(|| {
            format!("the ComfyUI workflow {name} saved no picture (add a SaveImage node)")
        })?;
        let path = self.fetch(file)?;
        Ok((path, workflow.name))
    }

    fn workflow(&self, name: &str) -> Result<Workflow, String> {
        Workflow::load(name, self.config.workflows.as_deref())
    }

    /// Queues `workflow` filled with the inputs and waits for it to finish.
    pub fn run(
        &self,
        workflow: &Workflow,
        prompt: Option<&str>,
        picture: Option<&Path>,
        seed: u64,
        stage: &str,
        progress: Progress<'_>,
    ) -> Result<Saved, String> {
        let image = match picture {
            Some(path) if workflow.uses("{{image}}") => Some(self.upload(path)?),
            _ => None,
        };
        let graph = workflow.fill(prompt, image.as_deref(), seed)?;
        let id = self.queue(&graph, &workflow.name)?;
        tracing::info!(%id, workflow = %workflow.name, "ComfyUI job queued");
        let started = Instant::now();
        loop {
            if let Some(entry) = self.history(&id)? {
                return outcome(&entry, &workflow.name);
            }
            if started.elapsed() > self.job_timeout {
                let _ = self
                    .agent
                    .post(format!("{}/api/jobs/{id}/cancel", self.base))
                    .send_empty();
                return Err(format!(
                    "the ComfyUI workflow {} took longer than {} minutes and was cancelled",
                    workflow.name,
                    self.job_timeout.as_secs() / 60
                ));
            }
            let waited = started.elapsed().as_secs();
            progress(0.0, &format!("{stage} ({waited} s)"));
            std::thread::sleep(self.poll);
        }
    }

    fn queue(&self, graph: &Value, name: &str) -> Result<String, String> {
        let mut response = self
            .agent
            .post(format!("{}/prompt", self.base))
            .send_json(json!({"prompt": graph, "client_id": "rust-hub"}))
            .map_err(|e| self.unreachable(&e))?;
        let status = response.status();
        let body: Value = response.body_mut().read_json().unwrap_or(Value::Null);
        if !status.is_success() {
            return Err(format!(
                "ComfyUI refused the workflow {name}: {}",
                describe_refusal(&body)
            ));
        }
        body["prompt_id"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("ComfyUI accepted the workflow {name} but gave no job id"))
    }

    fn history(&self, id: &str) -> Result<Option<Value>, String> {
        let mut response = self
            .agent
            .get(format!("{}/history/{id}", self.base))
            .call()
            .map_err(|e| self.unreachable(&e))?;
        let body: Value = response
            .body_mut()
            .read_json()
            .map_err(|e| format!("unexpected answer from ComfyUI's history: {e}"))?;
        let entry = &body[id];
        let done = entry["status"]["completed"].as_bool() == Some(true)
            || entry["status"]["status_str"].as_str() == Some("error");
        Ok(done.then(|| entry.clone()))
    }

    /// Uploads a picture into ComfyUI's input folder and returns its name there.
    fn upload(&self, path: &Path) -> Result<String, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
        let name = format!("rust-hub-{}.{ext}", unique());
        let boundary = format!("rust-hub-{}", unique());
        let mut body = Vec::with_capacity(bytes.len() + 512);
        let mut field = |head: &str, data: &[u8]| {
            body.extend_from_slice(format!("--{boundary}\r\n{head}\r\n\r\n").as_bytes());
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        };
        field(
            &format!(
                "Content-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n\
                 Content-Type: application/octet-stream"
            ),
            &bytes,
        );
        field("Content-Disposition: form-data; name=\"type\"", b"input");
        field(
            "Content-Disposition: form-data; name=\"overwrite\"",
            b"true",
        );
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        let mut response = self
            .agent
            .post(format!("{}/upload/image", self.base))
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .send(&body[..])
            .map_err(|e| self.unreachable(&e))?;
        if !response.status().is_success() {
            return Err(format!(
                "ComfyUI refused the picture upload ({})",
                response.status()
            ));
        }
        let saved: Value = response
            .body_mut()
            .read_json()
            .map_err(|e| format!("unexpected answer from ComfyUI's upload: {e}"))?;
        let name = saved["name"].as_str().unwrap_or(&name);
        Ok(match saved["subfolder"].as_str() {
            Some(sub) if !sub.is_empty() => format!("{sub}/{name}"),
            _ => name.to_owned(),
        })
    }

    /// Downloads a saved file into the cache folder.
    pub fn fetch(&self, file: &SavedFile) -> Result<PathBuf, String> {
        let mut response = self
            .agent
            .get(format!("{}/view", self.base))
            .query("filename", &file.filename)
            .query("subfolder", &file.subfolder)
            .query("type", &file.kind)
            .call()
            .map_err(|e| self.unreachable(&e))?;
        if !response.status().is_success() {
            return Err(format!(
                "could not download {} from ComfyUI ({})",
                file.filename,
                response.status()
            ));
        }
        let bytes = response
            .body_mut()
            .with_config()
            .limit(MAX_DOWNLOAD)
            .read_to_vec()
            .map_err(|e| format!("download of {} failed: {e}", file.filename))?;
        std::fs::create_dir_all(&self.cache)
            .map_err(|e| format!("could not create {}: {e}", self.cache.display()))?;
        let local = Path::new(&file.filename)
            .file_name()
            .map_or_else(|| "output".into(), |n| n.to_string_lossy().into_owned());
        let path = self.cache.join(format!("{}-{local}", unique()));
        std::fs::write(&path, bytes)
            .map_err(|e| format!("could not write {}: {e}", path.display()))?;
        Ok(path)
    }

    fn unreachable(&self, e: &ureq::Error) -> String {
        format!("could not reach ComfyUI at {} ({e})", self.base)
    }
}

/// `comfyui` or `comfyui:<workflow>`: whether `model` names ComfyUI, and the
/// workflow if one is named.
pub fn parse_engine(model: &str) -> Option<Option<&str>> {
    let rest = model.strip_prefix("comfyui")?;
    match rest.strip_prefix(':') {
        Some(name) if !name.trim().is_empty() => Some(Some(name.trim())),
        Some(_) => Some(None),
        None if rest.is_empty() => Some(None),
        None => None,
    }
}

impl Generator for ComfyUi {
    fn describe(&self) -> String {
        format!("ComfyUI at {}", self.base)
    }

    /// A whole job in ComfyUI. With `model: "comfyui:<workflow>"` that one
    /// workflow does everything; otherwise the picture workflow turns the
    /// prompt into a picture (unless one was given) and the model workflow
    /// turns the picture into a model.
    fn generate(
        &self,
        request: &GenerateRequest,
        progress: Progress<'_>,
    ) -> Result<Generated, String> {
        let started = Instant::now();
        self.ensure_running(progress)?;
        let seed = request.seed.unwrap_or_else(random_seed);
        let prompt = request.prompt.as_deref().filter(|p| !p.trim().is_empty());
        let named = request.model.as_deref().and_then(parse_engine).flatten();
        let mut picture = request.image.clone();
        let mut image_model = None;

        let model_workflow = self.workflow(named.unwrap_or(&self.config.model_workflow))?;
        if model_workflow.makes() != Makes::Model {
            return Err(format!(
                "the ComfyUI workflow {} saves no 3D model; it needs a SaveGLB node",
                model_workflow.name
            ));
        }
        if picture.is_none() && !model_workflow.uses("{{prompt}}") {
            let prompt = prompt.ok_or("give a prompt or a picture")?;
            let picture_workflow = request
                .image_model
                .as_deref()
                .and_then(parse_engine)
                .flatten();
            let half = |f: f32, s: &str| progress(f * 0.4, s);
            let (path, name) = self.make_picture(prompt, seed, picture_workflow, &half)?;
            picture = Some(path);
            image_model = Some(format!("comfyui:{name}"));
        }
        let rest = |f: f32, s: &str| progress(0.4 + f * 0.6, s);
        let saved = self.run(
            &model_workflow,
            prompt,
            picture.as_deref(),
            seed,
            "making the 3D model in ComfyUI",
            &rest,
        );
        if self.config.free_after {
            self.free_memory();
        }
        let saved = saved?;
        let file = saved.models.last().ok_or_else(|| {
            format!(
                "the ComfyUI workflow {} saved no 3D model",
                model_workflow.name
            )
        })?;
        let path = self.fetch(file)?;
        progress(1.0, "done");
        Ok(Generated {
            path,
            info: GenerationInfo {
                model: format!("comfyui:{}", model_workflow.name),
                image_model,
                prompt: prompt.map(str::to_owned),
                seed,
                textured: false,
                seconds: started.elapsed().as_secs_f32(),
                image: picture,
            },
        })
    }
}

/// Turns a finished history entry into its saved files, or ComfyUI's error.
fn outcome(entry: &Value, name: &str) -> Result<Saved, String> {
    if entry["status"]["status_str"].as_str() == Some("error") {
        let detail = entry["status"]["messages"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|m| m[0] == "execution_error")
            .map_or_else(
                || "no details given".to_owned(),
                |m| {
                    format!(
                        "{} failed: {}",
                        m[1]["node_type"].as_str().unwrap_or("a node"),
                        m[1]["exception_message"].as_str().unwrap_or("").trim()
                    )
                },
            );
        return Err(format!("the ComfyUI workflow {name} failed: {detail}"));
    }
    let mut saved = Saved::default();
    let files = |list: &Value| -> Vec<SavedFile> {
        list.as_array()
            .into_iter()
            .flatten()
            .filter(|f| f["type"].as_str() != Some("temp"))
            .filter_map(|f| {
                Some(SavedFile {
                    filename: f["filename"].as_str()?.to_owned(),
                    subfolder: f["subfolder"].as_str().unwrap_or("").to_owned(),
                    kind: f["type"].as_str().unwrap_or("output").to_owned(),
                })
            })
            .collect()
    };
    for output in entry["outputs"]
        .as_object()
        .into_iter()
        .flat_map(|o| o.values())
    {
        saved.pictures.extend(files(&output["images"]));
        saved.models.extend(files(&output["3d"]));
        // Other 3D nodes report their files under other keys; keep any GLB.
        if let Some(map) = output.as_object() {
            for (key, list) in map {
                if key != "3d" && key != "images" {
                    saved.models.extend(
                        files(list)
                            .into_iter()
                            .filter(|f| f.filename.to_ascii_lowercase().ends_with(".glb")),
                    );
                }
            }
        }
    }
    Ok(saved)
}

/// ComfyUI's reasons for refusing a workflow, e.g. a model file that is missing.
fn describe_refusal(body: &Value) -> String {
    let mut reasons = Vec::new();
    if let Some(nodes) = body["node_errors"].as_object() {
        for (id, node) in nodes {
            let class = node["class_type"].as_str().unwrap_or("node");
            for e in node["errors"].as_array().into_iter().flatten() {
                let message = e["message"].as_str().unwrap_or("error");
                let details = e["details"].as_str().unwrap_or("");
                reasons.push(
                    format!("{class} (node {id}): {message} {details}")
                        .trim()
                        .to_owned(),
                );
            }
        }
    }
    if reasons.is_empty() {
        reasons.push(
            body["error"]["message"]
                .as_str()
                .unwrap_or("no reason given")
                .to_owned(),
        );
    }
    let mut text = reasons.join("; ");
    if text.contains("not in") || text.contains("Value not in list") {
        text.push_str(". A model file the workflow names is missing from ComfyUI's models folder");
    }
    if text.contains("does not exist") || text.contains("Cannot execute because node") {
        text.push_str(". A node type is missing; install its custom node pack in ComfyUI");
    }
    text
}

/// How to start a ComfyUI install.
#[derive(Debug, Clone, PartialEq)]
struct Launch {
    program: PathBuf,
    args: Vec<String>,
    cwd: PathBuf,
}

impl Launch {
    /// The portable build (`python_embeded`) or a checkout with `main.py`, in
    /// `dir` or one folder down (`ComfyUI_windows_portable`, `ComfyUI`).
    fn find(dir: &Path, port: u16) -> Option<Self> {
        let candidates = std::iter::once(dir.to_owned()).chain(
            std::fs::read_dir(dir)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_dir()),
        );
        let common = [
            "--listen".to_owned(),
            "127.0.0.1".to_owned(),
            "--port".to_owned(),
            port.to_string(),
        ];
        for d in candidates {
            let embedded = d.join("python_embeded").join("python.exe");
            if embedded.is_file() && d.join("ComfyUI").join("main.py").is_file() {
                let mut args = vec![
                    "-s".to_owned(),
                    "ComfyUI/main.py".to_owned(),
                    "--windows-standalone-build".to_owned(),
                ];
                args.extend(common.clone());
                return Some(Self {
                    program: embedded,
                    args,
                    cwd: d,
                });
            }
            if d.join("main.py").is_file() && d.join("comfy").is_dir() {
                let venv = ["venv", ".venv"]
                    .iter()
                    .flat_map(|v| {
                        [
                            d.join(v).join("Scripts").join("python.exe"),
                            d.join(v).join("bin").join("python"),
                        ]
                    })
                    .find(|p| p.is_file());
                let program = venv.unwrap_or_else(|| {
                    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
                });
                let mut args = vec!["main.py".to_owned()];
                args.extend(common.clone());
                return Some(Self {
                    program,
                    args,
                    cwd: d,
                });
            }
        }
        None
    }

    fn spawn(&self, log_dir: &Path) -> Result<std::process::Child, String> {
        let mut command = Command::new(&self.program);
        command
            .args(&self.args)
            .current_dir(&self.cwd)
            .stdin(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt as _;
            // Its own console window, as when started by hand, which also keeps
            // it running after the hub closes.
            const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;
            command.creation_flags(CREATE_NEW_CONSOLE);
        }
        #[cfg(not(windows))]
        {
            std::fs::create_dir_all(log_dir)
                .map_err(|e| format!("could not create {}: {e}", log_dir.display()))?;
            let log = std::fs::File::create(log_dir.join("comfyui.log"))
                .map_err(|e| format!("could not create the ComfyUI log: {e}"))?;
            let err = log.try_clone().map_err(|e| e.to_string())?;
            command.stdout(log).stderr(err);
        }
        let _ = log_dir;
        command.spawn().map_err(|e| {
            format!(
                "could not start ComfyUI with {}: {e}",
                self.program.display()
            )
        })
    }

    fn log_hint(&self, log_dir: &Path) -> String {
        if cfg!(windows) {
            format!(
                "its window showed the reason (folder {})",
                self.cwd.display()
            )
        } else {
            format!("its output is in {}", log_dir.join("comfyui.log").display())
        }
    }
}

/// A ComfyUI install in one of the usual places.
fn find_install() -> Option<PathBuf> {
    let home = std::env::home_dir();
    let mut places = vec![
        PathBuf::from(r"C:\ComfyUI"),
        PathBuf::from(r"C:\ComfyUI_windows_portable"),
    ];
    if let Some(h) = &home {
        places.extend([
            h.join("ComfyUI"),
            h.join("ComfyUI_windows_portable"),
            h.join("Documents").join("ComfyUI"),
        ]);
    }
    places
        .into_iter()
        .find(|p| p.is_dir() && Launch::find(p, 8188).is_some())
}

fn port_of(url: &str) -> u16 {
    url.rsplit_once(':')
        .and_then(|(_, rest)| rest.trim_end_matches('/').split('/').next()?.parse().ok())
        .unwrap_or(8188)
}

fn unique() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{nanos:x}")
}

/// A fresh seed for a request that names none.
pub fn random_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    // SplitMix64 of the clock: a different seed each run, within 32 bits so
    // every engine accepts it.
    let mut z = (nanos as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) & 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_names_parse() {
        assert_eq!(parse_engine("comfyui"), Some(None));
        assert_eq!(parse_engine("comfyui:"), Some(None));
        assert_eq!(parse_engine("comfyui:my-owl"), Some(Some("my-owl")));
        assert_eq!(parse_engine("comfyuix"), None);
        assert_eq!(parse_engine("hunyuan3d-2"), None);
    }

    #[test]
    fn ports_come_from_the_url() {
        assert_eq!(port_of("http://127.0.0.1:8188"), 8188);
        assert_eq!(port_of("http://box:9000/"), 9000);
        assert_eq!(port_of("http://localhost"), 8188);
    }

    #[test]
    fn installs_are_recognised() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Launch::find(dir.path(), 8188).is_none());
        let portable = dir.path().join("ComfyUI_windows_portable");
        std::fs::create_dir_all(portable.join("python_embeded")).unwrap();
        std::fs::create_dir_all(portable.join("ComfyUI")).unwrap();
        std::fs::write(portable.join("python_embeded/python.exe"), "").unwrap();
        std::fs::write(portable.join("ComfyUI/main.py"), "").unwrap();
        let launch = Launch::find(dir.path(), 8190).unwrap();
        assert_eq!(launch.cwd, portable);
        assert!(
            launch
                .args
                .contains(&"--windows-standalone-build".to_owned())
        );
        assert!(launch.args.contains(&"8190".to_owned()));
    }

    #[test]
    fn refusals_read_like_sentences() {
        let body = json!({
            "error": {"message": "Prompt outputs failed validation"},
            "node_errors": {"1": {"class_type": "ImageOnlyCheckpointLoader", "errors": [
                {"message": "Value not in list", "details": "ckpt_name: 'hunyuan_3d_v2.1.safetensors' not in []"}
            ]}}
        });
        let text = describe_refusal(&body);
        assert!(text.contains("ImageOnlyCheckpointLoader"), "{text}");
        assert!(
            text.contains("missing from ComfyUI's models folder"),
            "{text}"
        );
    }
}
