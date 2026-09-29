"""Load the portable weatherstrip USD and verify its runtime material profile."""

import dataclasses
import json
from pxr import Usd, UsdGeom
from .config import REPO_ROOT


def load_weatherstrip_points(config):
    path = REPO_ROOT / "assets/weatherstrip/weatherstrip.usda"
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError("Build assets first: scripts/python.sh scripts/build_asset.py")
    profile = json.loads(path.with_name("runtime.json").read_text())
    if profile["weatherstrip"] != dataclasses.asdict(config.weatherstrip):
        raise ValueError("USD material profile differs from config; rebuild the asset before running.")
    curve = UsdGeom.BasisCurves.Get(stage, "/Weatherstrip/Physics/Centerline")
    if curve.GetWrapAttr().Get() != "periodic":
        raise ValueError("Weatherstrip must be a closed periodic curve")
    points = [(float(p[0]), float(p[1]), float(p[2]) + config.rest_height_m) for p in curve.GetPointsAttr().Get()]
    if len(points) != config.weatherstrip.segments:
        raise ValueError("USD centerline count differs from material profile")
    return [*points, points[0]]
