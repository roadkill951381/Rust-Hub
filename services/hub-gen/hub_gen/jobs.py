"""A queue that runs one generation at a time on a worker thread (one GPU),
with progress, cancellation and pruning of old job folders."""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from hub_gen.pipeline import Pipeline
from hub_gen.progress import Cancelled
from hub_gen.schema import GenerateRequest, JobResult, JobView, Status

log = logging.getLogger(__name__)

FILES = ("mesh.glb", "reference.png", "object.png")


@dataclass
class Job:
    id: str
    request: GenerateRequest
    dir: Path
    status: Status = Status.queued
    progress: float = 0.0
    stage: str = "waiting for the GPU"
    error: str | None = None
    created: float = field(default_factory=time.time)
    result: JobResult | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def view(self) -> JobView:
        result = None
        if self.result is not None:
            result = self.result.model_copy(
                update={
                    "files": {
                        name: f"/v1/jobs/{self.id}/files/{name}" for name in self.result.files
                    }
                }
            )
        return JobView(
            id=self.id,
            status=self.status,
            progress=round(self.progress, 3),
            stage=self.stage,
            error=self.error,
            created=self.created,
            result=result,
        )


class JobQueue:
    def __init__(self, pipeline: Pipeline, output_dir: Path, keep: int = 50) -> None:
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.keep = keep
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self._worker is None:
            self._worker = threading.Thread(target=self._run, name="hub-gen-worker", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=5)
            self._worker = None

    def submit(self, request: GenerateRequest) -> Job:
        self.pipeline.validate(request)
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, request=request, dir=self.output_dir / job_id)
        with self._lock:
            self._jobs[job_id] = job
            self._prune()
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(reversed(self._jobs.values()))

    def waiting(self) -> int:
        return self._queue.qsize()

    def cancel(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is not None and job.status in (Status.queued, Status.running):
            job.cancel.set()
            if job.status == Status.queued:
                job.status = Status.cancelled
                job.stage = "cancelled"
        return job

    def file(self, job_id: str, name: str) -> Path | None:
        job = self.get(job_id)
        if job is None or job.result is None or name not in job.result.files or name not in FILES:
            return None
        path = job.dir / name
        return path if path.is_file() else None

    def _run(self) -> None:
        while (job := self._queue.get()) is not None:
            if job.cancel.is_set():
                continue
            job.status = Status.running
            job.dir.mkdir(parents=True, exist_ok=True)

            def report(fraction: float, stage: str, job: Job = job) -> None:
                if job.cancel.is_set():
                    raise Cancelled
                job.progress = max(job.progress, min(fraction, 1.0))
                job.stage = stage

            try:
                job.result = self.pipeline.run(job.request, job.dir, report)
                job.status = Status.done
                job.progress = 1.0
                job.stage = "done"
                log.info("job %s done in %.1f s", job.id, job.result.seconds)
            except Cancelled:
                job.status = Status.cancelled
                job.stage = "cancelled"
            except Exception as e:
                log.exception("job %s failed", job.id)
                job.status = Status.failed
                job.error = f"{type(e).__name__}: {e}"
                job.stage = "failed"

    def _prune(self) -> None:
        finished = [
            j for j in self._jobs.values() if j.status not in (Status.queued, Status.running)
        ]
        for job in finished[: max(0, len(self._jobs) - self.keep)]:
            del self._jobs[job.id]
            shutil.rmtree(job.dir, ignore_errors=True)
