"""The HTTP API's request and response bodies."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class GenerateRequest(BaseModel):
    prompt: str | None = Field(None, max_length=2000, description="What to make: one object.")
    image_b64: str | None = Field(
        None, description="A PNG or JPEG of one object, base64; used instead of the prompt."
    )
    model: str | None = Field(None, description="Shape model; the service default when omitted.")
    image_model: str | None = Field(None, description="Text-to-image model for the prompt.")
    texture: bool = Field(False, description="Also paint a colour texture.")
    seed: int | None = Field(None, ge=0, le=2**32 - 1)
    faces: int | None = Field(None, ge=500, le=5_000_000, description="Target triangle count.")
    steps: int | None = Field(None, ge=1, le=200, description="Override the model's step count.")

    @model_validator(mode="after")
    def needs_prompt_or_image(self) -> GenerateRequest:
        if not (self.prompt and self.prompt.strip()) and not self.image_b64:
            raise ValueError("give a prompt or an image")
        return self


class Status(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"
    cancelled = "cancelled"


class JobResult(BaseModel):
    model: str
    image_model: str | None = None
    prompt: str | None = None
    seed: int
    seconds: float
    textured: bool
    triangles: int
    vertices: int
    #: File name -> URL path, e.g. {"mesh.glb": "/v1/jobs/<id>/files/mesh.glb"}.
    files: dict[str, str]


class JobView(BaseModel):
    id: str
    status: Status
    progress: float
    stage: str
    error: str | None = None
    created: float
    result: JobResult | None = None


class ModelInfo(BaseModel):
    name: str
    kind: str
    available: bool
    loaded: bool
    vram_gb: float
    licence: str
    note: str


class Health(BaseModel):
    status: str
    version: str
    device: str
    default_model: str
    image_model: str
    texture_model: str
    queued: int
    models: list[ModelInfo]
