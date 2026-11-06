"""Clean-up applied to every generated mesh before it is returned."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

log = logging.getLogger(__name__)


def remove_floaters(mesh: trimesh.Trimesh, min_fraction: float = 0.01) -> trimesh.Trimesh:
    """Drops loose pieces with fewer than ``min_fraction`` of the largest piece's faces.

    Generators leave small specks around the main surface; slicers and game
    engines both do better without them.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    largest = max(len(p.faces) for p in parts)
    kept = [p for p in parts if len(p.faces) >= largest * min_fraction]
    log.info("removed %d floating pieces", len(parts) - len(kept))
    return trimesh.util.concatenate(kept)


def reduce_faces(mesh: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    """Quadric decimation down to ``faces`` triangles, if the mesh has more."""
    if faces <= 0 or len(mesh.faces) <= faces:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=faces)


def clean(mesh: trimesh.Trimesh, faces: int | None) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=True,  # merges duplicate vertices
    )
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh = remove_floaters(mesh)
    if faces:
        mesh = reduce_faces(mesh, faces)
    trimesh.repair.fix_normals(mesh)
    return mesh
