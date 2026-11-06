"""End to end through the HTTP API with the demo models (no GPU needed)."""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from hub_gen.api import create_app
from hub_gen.config import Settings
from hub_gen.imaging import frame_object, has_useful_alpha, remove_background


@pytest.fixture
def settings(tmp_path):
    s = Settings(output_dir=tmp_path / "jobs", token="secret")
    s.default_model, s.image_model, s.texture_model = "demo", "demo-image", "demo-texture"
    return s


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings), headers={"Authorization": "Bearer secret"}) as c:
        yield c


def wait(client, job_id, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def load_glb(data: bytes) -> trimesh.Trimesh:
    scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
    return trimesh.util.concatenate(list(scene.dump()))


def test_health_lists_models_and_requires_the_token(client, settings):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert {m["name"] for m in body["models"]} >= {"hunyuan3d-2", "triposg", "demo"}
    with TestClient(create_app(settings)) as anonymous:
        assert anonymous.get("/health").status_code == 401


def test_prompt_to_watertight_glb(client):
    job = client.post("/v1/jobs", json={"prompt": "a tall vase", "seed": 4}).json()
    assert job["status"] in ("queued", "running", "done")
    job = wait(client, job["id"])
    assert job["status"] == "done", job
    result = job["result"]
    assert result["seed"] == 4 and result["model"] == "demo"
    mesh = load_glb(client.get(result["files"]["mesh.glb"]).content)
    assert mesh.is_watertight and mesh.volume > 0
    assert len(mesh.faces) == result["triangles"]
    extents = mesh.bounding_box.extents
    assert extents[1] == pytest.approx(extents.max()), "the vase stands up along Y"
    png = client.get(result["files"]["reference.png"])
    assert png.headers["content-type"] == "image/png"


def test_same_seed_same_model(client):
    a = wait(client, client.post("/v1/jobs", json={"prompt": "cup", "seed": 9}).json()["id"])
    b = wait(client, client.post("/v1/jobs", json={"prompt": "cup", "seed": 9}).json()["id"])
    assert a["result"]["triangles"] == b["result"]["triangles"]
    ma = load_glb(client.get(a["result"]["files"]["mesh.glb"]).content)
    mb = load_glb(client.get(b["result"]["files"]["mesh.glb"]).content)
    assert np.allclose(ma.bounds, mb.bounds)


def test_image_input_texture_and_face_limit(client):
    picture = Image.new("RGB", (300, 400), (240, 240, 240))
    ImageDraw.Draw(picture).ellipse((80, 60, 220, 340), fill=(200, 40, 40))
    buffer = io.BytesIO()
    picture.save(buffer, "PNG")
    body = {
        "image_b64": base64.b64encode(buffer.getvalue()).decode(),
        "texture": True,
        "faces": 1000,
    }
    job = wait(client, client.post("/v1/jobs", json=body).json()["id"])
    assert job["status"] == "done", job
    assert job["result"]["textured"]
    assert job["result"]["triangles"] <= 1000


def test_bad_requests_fail_fast(client):
    assert client.post("/v1/jobs", json={}).status_code == 422
    r = client.post("/v1/jobs", json={"prompt": "x", "model": "nope"})
    assert r.status_code == 400 and "unknown shape model" in r.json()["detail"]
    r = client.post("/v1/jobs", json={"image_b64": "not base64!"})
    assert r.status_code == 400
    assert client.get("/v1/jobs/missing").status_code == 404
    assert client.get("/v1/jobs/missing/files/mesh.glb").status_code == 404


def test_cancel_marks_the_job(client):
    job = client.post("/v1/jobs", json={"prompt": "lamp"}).json()
    cancelled = client.delete(f"/v1/jobs/{job['id']}").json()
    assert cancelled["status"] in ("cancelled", "running", "done")
    final = wait(client, job["id"])
    assert final["status"] in ("cancelled", "done")


def test_framing_crops_and_centres():
    picture = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    ImageDraw.Draw(picture).rectangle((10, 10, 49, 89), fill=(0, 255, 0, 255))
    assert has_useful_alpha(picture)
    framed = frame_object(remove_background(picture), size=120, padding=0.1)
    alpha = np.asarray(framed.getchannel("A")) > 127
    rows, cols = np.flatnonzero(alpha.any(1)), np.flatnonzero(alpha.any(0))
    assert abs((rows[0] + rows[-1]) / 2 - 60) <= 1 and abs((cols[0] + cols[-1]) / 2 - 60) <= 1
    assert rows[-1] - rows[0] == pytest.approx(100, abs=2)


def test_versions_report_packages_and_cached_weights(client, tmp_path, monkeypatch):
    cache = tmp_path / "hf"
    snap = cache / "models--tencent--Hunyuan3D-2" / "snapshots" / "abc123"
    (snap / "hunyuan3d-dit-v2-0-turbo").mkdir(parents=True)
    (cache / "models--tencent--Hunyuan3D-2" / "refs").mkdir()
    (cache / "models--tencent--Hunyuan3D-2" / "refs" / "main").write_text("abc123")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HY3DGEN_MODELS", str(tmp_path / "none"))

    body = client.get("/v1/versions").json()
    names = {p["name"].lower(): p for p in body["packages"]}
    assert names["fastapi"]["installed"]
    assert names["fastapi"]["wanted"].startswith(">=")
    weights = {w["repo"]: w for w in body["weights"]}
    hunyuan = weights["tencent/Hunyuan3D-2"]
    assert hunyuan["revision"] == "abc123"
    assert hunyuan["downloaded"] == ["hunyuan3d-2"]
    assert set(hunyuan["models"]) == {"hunyuan3d-2", "hunyuan3d-paint"}
    assert weights["tencent/Hunyuan3D-2mini"]["revision"] is None
    assert body["project_dir"].endswith("hub-gen")


def test_extras_count_only_when_all_their_packages_are_installed(monkeypatch):
    from hub_gen import versions as v

    reqs = [
        {"name": "fastapi", "wanted": ">=0.115", "url": None, "extra": None, "conditional": False},
        # pydantic is installed, but the extra's other package is not.
        {"name": "pydantic", "wanted": "", "url": None, "extra": "flux", "conditional": False},
        {"name": "no-such-pkg", "wanted": "", "url": None, "extra": "flux", "conditional": False},
        # Only platform-specific packages missing: the extra still counts.
        {"name": "numpy", "wanted": "", "url": None, "extra": "hunyuan", "conditional": False},
        {
            "name": "no-such-gpu-pkg",
            "wanted": "",
            "url": None,
            "extra": "hunyuan",
            "conditional": True,
        },
    ]
    assert v.extras(reqs) == ["hunyuan"]
    names = {p["name"] for p in v.packages(reqs)}
    assert names == {"fastapi", "numpy"}, "flux's pydantic is not reported"


def test_only_missing_weights_are_announced_as_downloads(tmp_path, monkeypatch):
    from hub_gen import versions as v

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HY3DGEN_MODELS", str(tmp_path / "none"))
    assert v.needs_download("hunyuan3d-2")
    assert not v.needs_download("demo")
    repo = tmp_path / "models--tencent--Hunyuan3D-2"
    (repo / "snapshots" / "abc" / "hunyuan3d-dit-v2-0-turbo").mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc")
    assert not v.needs_download("hunyuan3d-2")
    assert v.needs_download("hunyuan3d-paint")
