"""Validate all three recorded surface tracks and USD dependencies."""

import argparse
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdUtils


def validate(path: Path) -> dict:
    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError(f"Cannot open recording: {path}")
    tracks = []
    for layer in range(3):
        mesh = UsdGeom.Mesh.Get(stage, f"/root/Weatherstrip_{layer}/Surface")
        if not mesh:
            raise ValueError(f"Missing surface track {layer}")
        times = mesh.GetPointsAttr().GetTimeSamples()
        if len(times) < 2:
            raise ValueError("Expected a recording with multiple time samples")
        for t in times:
            points = np.asarray(mesh.GetPointsAttr().Get(t))
            normals = np.asarray(mesh.GetNormalsAttr().Get(t))
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError(f"Invalid points at layer {layer}, time {t}")
            if normals.shape != points.shape or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=2e-4):
                raise ValueError(f"Invalid vertex normals at layer {layer}, time {t}")
        tracks.append({"layer": layer, "time_samples": len(times), "vertices": len(points)})
    _, _, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    if unresolved:
        raise ValueError(f"Unresolved USD dependencies: {unresolved}")
    return {"status": "passed", "recording": path.name, "tracks": tracks, "unresolved_dependencies": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.recording), indent=2))
