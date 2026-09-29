"""USD presentation driven by Newton body states; no animation drives physics."""

import math
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt


def rubber_material(stage, path="/root/Materials/EPDM"):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.008, 0.009, 0.011))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.78)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


RENDER_SUBDIVISIONS = 4
RADIAL_SIDES = 16


def smooth_centerline(centers, subdivisions=RENDER_SUBDIVISIONS):
    """Periodic Catmull-Rom skin: interpolate physical centers without moving them."""
    p0, p1 = np.roll(centers, 1, axis=0), centers
    p2, p3 = np.roll(centers, -1, axis=0), np.roll(centers, -2, axis=0)
    t = np.arange(subdivisions, dtype=np.float64)[None, :, None] / subdivisions
    return (
        0.5
        * (
            2 * p1[:, None, :]
            + (-p0 + p2)[:, None, :] * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3)[:, None, :] * t * t
            + (-p0 + 3 * p1 - 3 * p2 + p3)[:, None, :] * t * t * t
        )
    ).reshape(-1, 3)


def _transport(normal, previous, current):
    axis = np.cross(previous, current)
    cosine = float(np.clip(np.dot(previous, current), -1.0, 1.0))
    # Minimal rotation; avoid an arbitrary world-up switch near vertical tangents.
    if cosine > -0.999999:
        normal = normal + np.cross(axis, normal) + np.cross(axis, np.cross(axis, normal)) / (1 + cosine)
    normal = normal - current * np.dot(current, normal)
    length = np.linalg.norm(normal)
    if length < 1e-10:
        normal = np.cross(current, np.eye(3)[np.argmin(np.abs(current))])
        length = np.linalg.norm(normal)
    return normal / length


def tube_geometry(centers, radius, sides=RADIAL_SIDES):
    centers = smooth_centerline(np.asarray(centers, dtype=np.float64))
    tangent = np.roll(centers, -1, axis=0) - np.roll(centers, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-10)
    normals = np.empty_like(tangent)
    first = np.cross(tangent[0], np.eye(3)[np.argmin(np.abs(tangent[0]))])
    normals[0] = first / np.linalg.norm(first)
    for i in range(1, len(centers)):
        normals[i] = _transport(normals[i - 1], tangent[i - 1], tangent[i])
    closing = _transport(normals[-1], tangent[-1], tangent[0])
    twist = math.atan2(np.dot(np.cross(closing, normals[0]), tangent[0]), np.dot(closing, normals[0]))
    correction = np.arange(len(centers))[:, None] * (twist / len(centers))
    normals = normals * np.cos(correction) + np.cross(tangent, normals) * np.sin(correction)
    binormal = np.cross(tangent, normals)
    angle = np.arange(sides) * (2 * math.pi / sides)
    radial = normals[:, None, :] * np.cos(angle)[None, :, None] + binormal[:, None, :] * np.sin(angle)[None, :, None]
    points = (centers[:, None, :] + radius * radial).reshape(-1, 3).astype(np.float32)
    return points, radial.reshape(-1, 3).astype(np.float32)


def tube_points(centers, radius, sides=RADIAL_SIDES):
    return tube_geometry(centers, radius, sides)[0]


def create_tube(stage, centers, radius, path="/root/Weatherstrip/Surface"):
    mesh = UsdGeom.Mesh.Define(stage, path)
    points, normals = tube_geometry(centers, radius)
    count, sides = len(points) // RADIAL_SIDES, RADIAL_SIDES
    faces = []
    for i in range(count):
        for j in range(sides):
            faces.extend(
                [
                    i * sides + j,
                    i * sides + (j + 1) % sides,
                    ((i + 1) % count) * sides + (j + 1) % sides,
                    ((i + 1) % count) * sides + j,
                ]
            )
    mesh.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation("vertex")
    mesh.GetPrim().SetCustomDataByKey("render_centerline_samples", count)
    mesh.GetPrim().SetCustomDataByKey("physics_centerline_samples", len(centers))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(rubber_material(stage))
    return mesh


class Presentation:
    def __init__(self, viewer, sim):
        self.viewer, self.sim, self.stage = viewer, sim, viewer.stage
        from .skin import TubeSkin

        body_q = sim.state_0.body_q.numpy()
        self.meshes = [
            create_tube(
                self.stage,
                body_q[strip["bodies"], :3],
                sim.config.weatherstrip.cross_section_radius_m,
                f"/root/Weatherstrip_{i}/Surface",
            )
            for i, strip in enumerate(sim.strips)
        ]
        self.skin = TubeSkin(
            [strip["bodies"] for strip in sim.strips], sim.config.weatherstrip.cross_section_radius_m, sim.device
        )
        dome = UsdLux.DomeLight.Define(self.stage, "/root/Lighting/Dome")
        dome.CreateIntensityAttr(550.0)
        key = UsdLux.DistantLight.Define(self.stage, "/root/Lighting/Key")
        key.CreateIntensityAttr(2200.0)
        key.CreateAngleAttr(12.0)
        UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-35, -25, -20))
        self.stage.SetEndTimeCode(sim.config.full_cycle_frames)
        self.stage.GetDefaultPrim().SetCustomDataByKey(
            "weatherstrip:physics", "Newton 1.5 SolverCoupledADMM(MuJoCo,VBD); live physical finger contact"
        )
        self.stage.GetDefaultPrim().SetCustomDataByKey(
            "weatherstrip:calibration", "Qualitative EPDM-like rod; uncalibrated material, no certification claim"
        )
        self.update()

    def update(self):
        surfaces = self.skin.update(self.sim.state_0.body_q)
        with Sdf.ChangeBlock():
            for mesh, data in zip(self.meshes, surfaces):
                mesh.GetPointsAttr().Set(
                    Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(data[:, 0])), self.viewer._frame_index
                )
                mesh.GetNormalsAttr().Set(
                    Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(data[:, 1])), self.viewer._frame_index
                )
