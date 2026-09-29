"""Validate USD structure, geometry, physical import and exact runtime profile."""

import argparse
import json
import numpy as np
from pxr import Usd, UsdGeom, UsdUtils
import newton
from newton_weatherstrip.config import REPO_ROOT, load_config
from newton_weatherstrip.asset import load_weatherstrip_points

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default=str(REPO_ROOT / "config/default.toml"))
args = parser.parse_args()
cfg = load_config(args.config, require_asset=False)
path = REPO_ROOT / "assets/weatherstrip/weatherstrip.usda"
s = Usd.Stage.Open(str(path))
assert s.GetDefaultPrim().GetPath() == "/Weatherstrip"
assert UsdGeom.GetStageMetersPerUnit(s) == 1.0
assert UsdGeom.GetStageUpAxis(s) == "Z"
points = np.asarray(load_weatherstrip_points(cfg))
assert np.isfinite(points).all() and np.allclose(points[0], points[-1])
assert np.allclose(
    np.ptp(points, axis=0)[:2], [2 * cfg.weatherstrip.rest_radius_m, 2 * cfg.weatherstrip.minor_radius_m]
)
mesh = UsdGeom.Mesh.Get(s, "/Weatherstrip/Visual/Surface")
assert sum(mesh.GetFaceVertexCountsAttr().Get()) == len(mesh.GetFaceVertexIndicesAttr().Get())
assert max(mesh.GetFaceVertexIndicesAttr().Get()) < len(mesh.GetPointsAttr().Get())
normals = np.asarray(mesh.GetNormalsAttr().Get())
assert mesh.GetNormalsInterpolation() == "vertex"
assert len(normals) == len(mesh.GetPointsAttr().Get()) == cfg.weatherstrip.segments * 4 * 16
assert np.isfinite(normals).all()
assert np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5)
_, _, unresolved = UsdUtils.ComputeAllDependencies(str(path))
assert not unresolved, unresolved
b = newton.ModelBuilder()
result = b.add_usd(str(path), return_deformable_results=True)
bodies, joints = result["path_cable_map"]["/Weatherstrip/Physics/Centerline"]
assert len(bodies) == len(joints) == cfg.weatherstrip.segments
mass = sum(b.body_mass[i] for i in bodies)
assert abs(mass - cfg.weatherstrip.total_mass_kg) < 1e-5
shape_ids = {body: next(i for i, v in enumerate(b.shape_body) if v == body) for body in bodies}
pair_set = {tuple(sorted(p)) for p in b.shape_collision_filter_pairs}
assert tuple(sorted((shape_ids[bodies[0]], shape_ids[bodies[2]]))) in pair_set
assert tuple(sorted((shape_ids[bodies[0]], shape_ids[bodies[10]]))) not in pair_set
report = {
    "status": "passed",
    "default_prim": "/Weatherstrip",
    "meters_per_unit": 1,
    "up_axis": "Z",
    "segments": len(bodies),
    "closed_cable_joints": len(joints),
    "imported_mass_kg": mass,
    "unresolved_dependencies": list(unresolved),
    "newton_usd_deformable_import": True,
    "neighbor_collision_exclusions": True,
    "material_calibrated": False,
    "formal_simready_certification": False,
}
(REPO_ROOT / "results").mkdir(exist_ok=True)
(REPO_ROOT / "results/asset_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
