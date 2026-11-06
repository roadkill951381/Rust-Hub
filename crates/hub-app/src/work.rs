//! Work that runs off the UI thread: long commands (generate, hollow, Blender
//! jobs) and print checks, so the window keeps drawing while they run.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex, PoisonError};

use hub_core::analysis::{CheckOptions, PrintReport, check_printable_with};
use hub_core::scene::Object;
use hub_core::{Error, Job, JobOutput, ObjectId};

/// A job running on its own thread. The session commits its result with
/// [`Session::finish`](hub_core::Session::finish) once [`Running::poll`]
/// returns it.
pub struct Running {
    label: &'static str,
    progress: Arc<Mutex<(f32, String)>>,
    result: Receiver<hub_core::Result<JobOutput>>,
}

impl Running {
    pub fn start(job: Job, ctx: egui::Context) -> Self {
        let label = job.label();
        let progress = Arc::new(Mutex::new((0.0, format!("{label}…"))));
        let (tx, result) = mpsc::channel();
        let shared = Arc::clone(&progress);
        std::thread::Builder::new()
            .name(format!("job: {label}"))
            .spawn(move || {
                let report = |fraction: f32, stage: &str| {
                    *shared.lock().unwrap_or_else(PoisonError::into_inner) =
                        (fraction.clamp(0.0, 1.0), stage.to_owned());
                    ctx.request_repaint();
                };
                let output = job.run(&report);
                // The receiver is gone only if the app closed; nothing to do then.
                let _ = tx.send(output);
                ctx.request_repaint();
            })
            .expect("spawn job thread");
        Self {
            label,
            progress,
            result,
        }
    }

    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Fraction done and the current stage.
    pub fn progress(&self) -> (f32, String) {
        self.progress
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The result, once the job has finished.
    pub fn poll(&self) -> Option<hub_core::Result<JobOutput>> {
        match self.result.try_recv() {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err(Error::Engine(format!(
                "{} stopped unexpectedly",
                self.label
            )))),
        }
    }
}

/// What a print report was computed for: session revision, object and the
/// wall threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReportKey {
    pub revision: u64,
    pub object: ObjectId,
    pub min_wall_mm: f32,
}

/// The print report for the selected object, recomputed in the background
/// whenever the key changes. Wall thickness on a large generated model takes
/// a moment, too long to do while drawing a frame.
#[derive(Default)]
pub struct ReportCache {
    current: Option<(ReportKey, PrintReport)>,
    pending: Option<(ReportKey, Receiver<PrintReport>)>,
}

impl ReportCache {
    /// The newest finished report for `key`'s object, which may be from an
    /// earlier revision while a fresh one is computed, plus whether it is
    /// up to date.
    pub fn get(
        &mut self,
        key: ReportKey,
        object: &Object,
        ctx: &egui::Context,
    ) -> Option<(&PrintReport, bool)> {
        if let Some((pending_key, rx)) = &self.pending {
            match rx.try_recv() {
                Ok(report) => {
                    self.current = Some((*pending_key, report));
                    self.pending = None;
                }
                Err(TryRecvError::Disconnected) => self.pending = None,
                Err(TryRecvError::Empty) => {}
            }
        }
        let fresh = self.current.as_ref().is_some_and(|(k, _)| *k == key);
        let computing = self.pending.as_ref().is_some_and(|(k, _)| *k == key);
        if !fresh && !computing {
            let (tx, rx) = mpsc::channel();
            let object = object.clone();
            let ctx = ctx.clone();
            let options = CheckOptions {
                min_wall_mm: key.min_wall_mm,
            };
            std::thread::spawn(move || {
                let _ = tx.send(check_printable_with(&object, &options));
                ctx.request_repaint();
            });
            self.pending = Some((key, rx));
        }
        self.current
            .as_ref()
            .filter(|(k, _)| k.object == key.object)
            .map(|(k, report)| (report, *k == key))
    }
}
