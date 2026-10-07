"""Zone-id map for the generated square-with-hole Fluent mesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from multipinn.geometry.geometry import Geometry
from multipinn.mesh.grid_reader import Face, FaceType, Grid

ZONE_INTERIOR = 2
ZONE_OUTER = 3
ZONE_HOLE = 4

ZONE_NAMES = {
    ZONE_INTERIOR: "interior",
    ZONE_OUTER: "outer_wall",
    ZONE_HOLE: "hole_wall",
}

MESH_FILENAME = "square_hole.msh"


def mesh_path(example_dir: Path | None = None) -> Path:
    root = example_dir or Path(__file__).resolve().parent
    return root / "meshes" / MESH_FILENAME


class MeshZoneGeometry(Geometry):
    """Collocation geometry from one Fluent face zone, no ``/50`` rescaling."""

    def __init__(self, face: Face):
        pts = np.asarray(
            [c.middle_point for c in face.connections], dtype=np.float32
        )
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError(f"face zone {face.zone_id} has no connections")
        lo = pts.min(axis=0).astype(np.float32)
        hi = pts.max(axis=0).astype(np.float32)
        super().__init__(pts.shape[1], (lo, hi), float(np.linalg.norm(hi - lo)))
        self.face = face
        self.zone_id = face.zone_id
        self.points = torch.tensor(pts, dtype=torch.float32)
        self._pts = pts
        if 0 in face.connections[0].cells:
            nrm = np.asarray([c.normal for c in face.connections], dtype=np.float32)
            self.points.normals = torch.tensor(nrm, dtype=torch.float32)
            self.normals = nrm
        else:
            self.normals = None

    def random_points(self, n, random="pseudo"):
        n_have = int(self._pts.shape[0])
        replace = n > n_have
        idx = np.random.choice(n_have, size=n, replace=replace)
        return self._pts[idx].astype(np.float32)


def geometries_from_grid(grid: Grid) -> dict[str, MeshZoneGeometry]:
    missing = []
    faces = {}
    for zid, name in ZONE_NAMES.items():
        face = grid.get_face_by_id(zid)
        if face is None:
            missing.append(zid)
            continue
        faces[name] = MeshZoneGeometry(face)
    if missing:
        raise KeyError(f"mesh is missing zone-id(s) {missing}")
    return faces


def require_wall_zones(grid: Grid) -> None:
    interior = grid.get_face_by_id(ZONE_INTERIOR)
    outer = grid.get_face_by_id(ZONE_OUTER)
    hole = grid.get_face_by_id(ZONE_HOLE)
    if interior is None or outer is None or hole is None:
        raise KeyError("expected zone-ids 2 (interior), 3 (outer), 4 (hole)")
    if interior.type != FaceType.INTERIOR:
        raise ValueError(f"zone 2 type is {interior.type}, expected INTERIOR")
    if outer.type != FaceType.WALL or hole.type != FaceType.WALL:
        raise ValueError("zones 3 and 4 must be WALL")
