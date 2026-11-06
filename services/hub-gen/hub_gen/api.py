"""The HTTP API the hub calls. JSON in and out; files served per job."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse

from hub_gen import __version__
from hub_gen.backends.base import Models
from hub_gen.config import Settings
from hub_gen.jobs import JobQueue
from hub_gen.pipeline import BadRequest, Pipeline
from hub_gen.schema import GenerateRequest, Health, JobView, ModelInfo


def create_app(settings: Settings) -> FastAPI:
    models = Models(settings)
    jobs = JobQueue(
        Pipeline(settings, models),
        settings.output_dir,
        settings.keep_jobs,
        idle_unload=settings.idle_unload,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        jobs.start()
        yield
        jobs.stop()

    def authorize(request: Request) -> None:
        if settings.token is None:
            return
        header = request.headers.get("authorization", "")
        given = header.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(given.encode(), settings.token.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or wrong bearer token")

    app = FastAPI(
        title="hub-gen",
        version=__version__,
        description="Local 3D generation for Rust-Hub.",
        lifespan=lifespan,
        dependencies=[Depends(authorize)],
    )
    app.state.jobs = jobs
    app.state.models = models

    @app.get("/health")
    def health() -> Health:
        return Health(
            status="ok",
            version=__version__,
            device=_describe_device(settings.device),
            default_model=settings.default_model,
            image_model=settings.image_model,
            multiview_model=settings.multiview_model,
            texture_model=settings.texture_model,
            queued=jobs.waiting(),
            models=[ModelInfo(**m) for m in models.status()],
        )

    @app.get("/v1/versions")
    def versions() -> dict[str, Any]:
        """Packages and cached model weights, for the hub's update check."""
        from hub_gen import versions as v

        return v.report()

    @app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit(body: GenerateRequest) -> JobView:
        try:
            return jobs.submit(body).view()
        except BadRequest as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

    @app.get("/v1/jobs")
    def list_jobs() -> list[JobView]:
        return [j.view() for j in jobs.list()]

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> JobView:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job {job_id}")
        return job.view()

    @app.delete("/v1/jobs/{job_id}")
    def cancel_job(job_id: str) -> JobView:
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job {job_id}")
        return job.view()

    @app.get("/v1/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str) -> FileResponse:
        path = jobs.file(job_id, name)
        if path is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no file {name} for job {job_id}")
        media = "model/gltf-binary" if name.endswith(".glb") else "image/png"
        return FileResponse(path, media_type=media, filename=name)

    return app


def _describe_device(device: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu (PyTorch not installed: demo models only)"
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device))
        return f"{device} {props.name} {props.total_memory / 2**30:.0f} GB"
    return f"{device} (CUDA unavailable)" if device.startswith("cuda") else device
