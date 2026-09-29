"""Read-only project, SimReady asset, and Newton import preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_report(stage, expected_variant: str) -> dict[str, object]:
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("USD has no default prim")
    physics = default_prim.GetVariantSets().GetVariantSet("Physics")
    used_layers = [Path(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath]
    return {
        "default_prim": str(default_prim.GetPath()),
        "physics_variants": list(physics.GetVariantNames()),
        "selected_physics_variant_before_runtime": physics.GetVariantSelection(),
        "required_physics_variant": expected_variant,
        "simready_profile": (
            stage.GetRootLayer().customLayerData.get("SimReady_Metadata", {}).get("validation", {}).get("profile")
        ),
        "used_layer_count": len(used_layers),
        "missing_layers": [str(path) for path in used_layers if not path.is_file()],
    }


def _find_suffix(mapping: dict[str, int], suffix: str) -> int:
    matches = [index for path, index in mapping.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one body ending in {suffix!r}, got {matches}")
    return matches[0]


def run_doctor(config_path: str | Path, *, runtime: bool, import_smoke: bool, device: str) -> dict[str, object]:
    config = load_config(config_path)
    report: dict[str, object] = {
        "schema_version": 2,
        "config": str(config.source_path),
        "config_valid": True,
        "robot_usd": str(config.assets.robot_usd),
        "robot_usd_sha256": _sha256(config.assets.robot_usd),
        "gripper_usd": str(config.assets.gripper_usd),
        "gripper_usd_sha256": _sha256(config.assets.gripper_usd),
        "full_cycle_frames": config.full_cycle_frames,
        "full_cycle_duration_s": config.trajectory.total_duration_s,
        "weatherstrip_model": "closed Newton VBD rod",
        "weatherstrip_density_kg_m3": config.weatherstrip.rod_density_kg_m3,
        "runtime_checked": False,
        "import_smoke_checked": False,
    }
    if not runtime and not import_smoke:
        return report

    import newton
    import warp as wp
    from newton.solvers import SolverMuJoCo
    from pxr import Usd

    report["runtime_checked"] = True
    report["newton_version"] = getattr(newton, "__version__", "unknown")
    report["warp_version"] = getattr(wp, "__version__", "unknown")
    report["available_devices"] = [str(found) for found in wp.get_devices()]

    robot_stage = Usd.Stage.Open(str(config.assets.robot_usd))
    gripper_stage = Usd.Stage.Open(str(config.assets.gripper_usd))
    if robot_stage is None or gripper_stage is None:
        raise RuntimeError("Could not open robot or gripper USD")

    robot_report = _stage_report(robot_stage, config.assets.physics_variant)
    gripper_report = _stage_report(gripper_stage, config.assets.gripper_physics_variant)
    report["robot"] = robot_report
    report["gripper"] = gripper_report

    if robot_report["simready_profile"] != config.assets.simready_profile:
        raise RuntimeError(
            f"Expected SimReady profile {config.assets.simready_profile!r}, got {robot_report['simready_profile']!r}"
        )
    for name, stage_report in (("robot", robot_report), ("gripper", gripper_report)):
        if stage_report["missing_layers"]:
            raise RuntimeError(f"{name} USD has missing composed layers: {stage_report['missing_layers']}")
        if stage_report["required_physics_variant"] not in stage_report["physics_variants"]:
            raise RuntimeError(f"{name} USD lacks Physics={stage_report['required_physics_variant']!r}")

    if import_smoke:
        robot_stage.GetDefaultPrim().GetVariantSets().SetSelection("Physics", config.assets.physics_variant)
        gripper_stage.GetDefaultPrim().GetVariantSets().SetSelection("Physics", config.assets.gripper_physics_variant)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        half_spacing = 0.5 * config.scene.robot_spacing_m
        for x_m, yaw_rad in ((-half_spacing, 0.0), (half_spacing, math.pi)):
            robot_result = builder.add_usd(
                robot_stage,
                xform=wp.transform(
                    (x_m, 0.0, config.scene.robot_base_z_m),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw_rad),
                ),
                floating=False,
                override_root_xform=True,
                root_path=config.assets.robot_prim,
                enable_self_collisions=False,
                load_static_visual_shapes=False,
                hide_collision_shapes=True,
            )
            ee_body = _find_suffix(robot_result["path_body_map"], "/J6_link")
            builder.add_usd(
                gripper_stage,
                xform=wp.transform(wp.vec3(0.160, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)),
                parent_body=ee_body,
                override_root_xform=True,
                floating=False,
                root_path=config.assets.gripper_prim,
                enable_self_collisions=False,
                load_static_visual_shapes=False,
                hide_collision_shapes=True,
            )

        import_counts = {
            "bodies": builder.body_count,
            "joints": builder.joint_count,
            "coordinates": builder.joint_coord_count,
            "shapes": builder.shape_count,
            "mimic_constraints": len(builder.constraint_mimic_joint0),
        }
        expected = {
            "bodies": 32,
            "joints": 32,
            "coordinates": 24,
            "mimic_constraints": 10,
        }
        if any(import_counts[k] != v for k, v in expected.items()):
            raise RuntimeError(f"Unexpected dual-assembly import counts: {import_counts} != {expected}")
        model = builder.finalize(device=device)
        report["import_smoke_checked"] = True
        report["import_counts"] = import_counts
        report["runtime_added_fingertip_pad_shapes"] = 0
        report["finalized_device"] = str(model.device)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--import-smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_doctor(
        args.config,
        runtime=args.runtime or args.import_smoke,
        import_smoke=args.import_smoke,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
