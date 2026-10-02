"""Author the reusable USD curve deformable plus calibrated-unit runtime profile.

PhysicsCurvesDeformableSimAPI is a proposal schema supported by this pinned
Newton importer; support in arbitrary USD viewers is not implied.
"""

import dataclasses
import hashlib
import argparse
import json
import math
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from newton_weatherstrip.config import load_config, REPO_ROOT
from newton_weatherstrip.geometry import ring_points
from newton_weatherstrip.presentation import create_tube

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default=str(REPO_ROOT / "config/default.toml"))
args = parser.parse_args()
cfg = load_config(args.config, require_asset=False)
w = cfg.weatherstrip
out = REPO_ROOT / "assets/weatherstrip"
out.mkdir(exist_ok=True)
stage = Usd.Stage.CreateNew(str(out / "weatherstrip.usda"))
root = UsdGeom.Xform.Define(stage, "/Weatherstrip")
stage.SetDefaultPrim(root.GetPrim())
UsdGeom.SetStageUpAxis(stage, "Z")
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root.GetPrim().SetCustomData(
    {
        "asset_name": "Oval EPDM-like weatherstrip",
        "material_calibrated": False,
        "physics_backend": "Newton VBD closed Cosserat rod",
        "conformance": "Simulation-ready example; not formally SimReady certified",
        "schema_support": "Pinned Newton 1.6 deformable proposal importer",
    }
)
points = np.asarray(ring_points(w.segments, w.rest_radius_m, 0.0, w.minor_radius_m), dtype=np.float32)
curve = UsdGeom.BasisCurves.Define(stage, "/Weatherstrip/Physics/Centerline")
curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
curve.CreateCurveVertexCountsAttr([w.segments])
curve.CreateTypeAttr("linear")
curve.CreateWrapAttr("periodic")
curve.CreateWidthsAttr([2 * w.cross_section_radius_m])
curve.SetWidthsInterpolation("constant")
curve.CreatePurposeAttr("guide")
curve.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")
curve.GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
UsdPhysics.CollisionAPI.Apply(curve.GetPrim()).CreateCollisionEnabledAttr(True)
# Curve moduli are Pa (not per-joint N/m). This matches the importer's mean-length conversion.
mean_l = float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).mean())
r = w.cross_section_radius_m
area, moment, polar = math.pi * r * r, math.pi * r**4 / 4, math.pi * r**4 / 2
mat = UsdShade.Material.Define(stage, "/Weatherstrip/Looks/RubberPhysics")
mat.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableMaterialAPI")
phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
phys.CreateStaticFrictionAttr(w.friction)
phys.CreateDynamicFrictionAttr(w.friction)
phys.CreateRestitutionAttr(0.0)
phys.CreateDensityAttr(w.rod_density_kg_m3)
for name, value in {
    "thickness": 2 * r,
    "stretchStiffness": w.stretch_stiffness_n_m * mean_l / area,
    "shearStiffness": w.shear_stiffness_n_m * mean_l / area,
    "bendStiffness": w.bend_stiffness_n_m * mean_l / moment,
    "twistStiffness": w.twist_stiffness_n_m * mean_l / polar,
}.items():
    mat.GetPrim().CreateAttribute("physics:" + name, Sdf.ValueTypeNames.Float).Set(value)
UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(mat, materialPurpose="physics")
UsdPhysics.MassAPI.Apply(curve.GetPrim()).CreateMassAttr(w.total_mass_kg)
for name, value in dataclasses.asdict(w).items():
    root.GetPrim().CreateAttribute("weatherstrip:" + name, Sdf.ValueTypeNames.Double).Set(float(value))
mesh = create_tube(stage, points, r, "/Weatherstrip/Visual/Surface")
# Keep all asset-local materials under the default prim for referencing.
from newton_weatherstrip.presentation import rubber_material

UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(rubber_material(stage, "/Weatherstrip/Looks/RubberSurface"))
stage.RemovePrim("/root")
for side, x in [("Left", -w.rest_radius_m), ("Right", w.rest_radius_m)]:
    p = UsdGeom.Xform.Define(stage, "/Weatherstrip/GraspSites/" + side)
    p.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
    p.GetPrim().SetCustomDataByKey("role", "two-finger grasp target; no attachment constraint")
root.GetPrim().CreateAttribute("weatherstrip:runtimeProfile", Sdf.ValueTypeNames.Asset).Set(
    Sdf.AssetPath("runtime.json")
)
profile = {
    "schema_version": 1,
    "backend": "Newton 1.6 SolverVBD",
    "units": "SI",
    "material_calibrated": False,
    "weatherstrip": dataclasses.asdict(w),
    "assumptions": [
        "Circular solid beam surrogate for weatherstrip cross section",
        "Elastic bend/shear/stretch/twist and viscous damping",
        "No measured EPDM hysteresis, compression or aging response",
    ],
    "adapter": "../../src/newton_weatherstrip/simulation.py",
}
(out / "runtime.json").write_text(json.dumps(profile, indent=2) + "\n")
# Preserve continuous-material neighborhood exclusions for standalone USD import.
filt = stage.DefinePrim("/Weatherstrip/Physics/NeighborExclusions", "PhysicsElementCollisionFilter")
for name in ["src0", "src1"]:
    filt.CreateRelationship("physics:" + name).SetTargets([curve.GetPath()])
pairs = [(i, (i + d) % w.segments) for i in range(w.segments) for d in (1, 2)]
filt.CreateAttribute("physics:filterEnabled", Sdf.ValueTypeNames.Bool).Set(True)
for side in (0, 1):
    filt.CreateAttribute(f"physics:groupElemIndices{side}", Sdf.ValueTypeNames.IntArray).Set([p[side] for p in pairs])
    filt.CreateAttribute(f"physics:groupElemCounts{side}", Sdf.ValueTypeNames.IntArray).Set([1] * len(pairs))
stage.GetRootLayer().Save()
# Keep generated ASCII USD diffs clean and reproducible.
usda = out / "weatherstrip.usda"
usda.write_text(usda.read_text().rstrip() + "\n")
print(out / "weatherstrip.usda")
