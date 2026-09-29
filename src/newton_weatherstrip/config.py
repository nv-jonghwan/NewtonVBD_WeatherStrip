"""Configuration loading and cross-field validation."""

from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.toml"


@dataclass(frozen=True)
class AssetConfig:
    robot_usd: Path
    robot_prim: str
    physics_variant: str
    simready_profile: str
    gripper_usd: Path
    gripper_prim: str
    gripper_physics_variant: str
    gripper_grasp_offset_m: float
    gripper_open_angle_rad: float
    gripper_hold_angle_rad: float


@dataclass(frozen=True)
class SceneConfig:
    robot_spacing_m: float
    robot_base_z_m: float
    table_half_extents_m: tuple[float, float, float]
    table_center_z_m: float
    table_friction: float
    stack_count: int
    stack_pitch_m: float

    @property
    def table_top_z_m(self) -> float:
        return self.table_center_z_m + self.table_half_extents_m[2]


@dataclass(frozen=True)
class WeatherStripConfig:
    segments: int
    minor_radius_m: float
    rest_radius_m: float
    cross_section_radius_m: float
    total_mass_kg: float
    stretch_stiffness_n_m: float
    stretch_damping_n_s_m: float
    shear_stiffness_n_m: float
    shear_damping_n_s_m: float
    bend_stiffness_n_m: float
    bend_damping_n_m_s: float
    twist_stiffness_n_m: float
    twist_damping_n_m_s: float
    friction: float

    @property
    def neighbor_chord_m(self) -> float:
        return 2.0 * min(self.rest_radius_m, self.minor_radius_m) * math.sin(math.pi / self.segments)

    @property
    def capsule_volume_m3(self) -> float:
        radius = self.cross_section_radius_m
        return math.pi * radius * radius * self.neighbor_chord_m + 4.0 * math.pi * radius**3 / 3.0

    @property
    def rod_density_kg_m3(self) -> float:
        from .geometry import closed_ring_points

        points = closed_ring_points(self.segments, self.rest_radius_m, 0.0, self.minor_radius_m)
        length = sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))
        r = self.cross_section_radius_m
        volume = math.pi * r * r * length  # add_rod uses cylinder mass, excluding capsule end caps
        return self.total_mass_kg / volume


@dataclass(frozen=True)
class SimulationConfig:
    fps: int
    substeps: int
    solver_iterations: int
    ik_iterations: int
    admm_iterations: int
    mujoco_iterations: int
    gravity_m_s2: float
    contact_stiffness_n_m: float
    contact_damping_n_s_m: float
    gripper_contact_stiffness_n_m: float
    gripper_contact_damping_n_s_m: float
    gripper_drive_stiffness_nm_rad: float
    gripper_drive_damping_nm_s_rad: float
    gripper_max_torque_nm: float
    device: str


@dataclass(frozen=True)
class TrajectoryConfig:
    ready_s: float
    descend_s: float
    grasp_s: float
    lift_s: float
    stretch_s: float
    hold_s: float
    relax_s: float
    sag_s: float
    release_s: float
    fall_s: float
    retract_s: float
    settle_s: float
    approach_height_m: float
    grip_depth_m: float
    sag_lowering_m: float
    clearance_m: float
    lift_height_m: float
    stretch_extension_m: float

    @property
    def durations_s(self) -> tuple[float, ...]:
        return (
            self.ready_s,
            self.descend_s,
            self.grasp_s,
            self.lift_s,
            self.stretch_s,
            self.hold_s,
            self.relax_s,
            self.sag_s,
            self.release_s,
            self.fall_s,
            self.retract_s,
            self.settle_s,
        )

    @property
    def total_duration_s(self) -> float:
        return sum(self.durations_s)


@dataclass(frozen=True)
class ProjectConfig:
    assets: AssetConfig
    scene: SceneConfig
    weatherstrip: WeatherStripConfig
    simulation: SimulationConfig
    trajectory: TrajectoryConfig
    source_path: Path

    @property
    def rest_height_m(self) -> float:
        return self.scene.table_top_z_m + self.weatherstrip.cross_section_radius_m

    @property
    def top_rest_height_m(self) -> float:
        return self.rest_height_m + (self.scene.stack_count - 1) * self.scene.stack_pitch_m

    @property
    def full_cycle_frames(self) -> int:
        return math.ceil(self.trajectory.total_duration_s * self.simulation.fps) + 1

    def validate(self, *, require_asset: bool = True) -> None:
        errors: list[str] = []
        ws = self.weatherstrip
        scene = self.scene
        sim = self.simulation
        traj = self.trajectory
        assets = self.assets

        if scene.stack_count != 3 or scene.stack_pitch_m < 2 * ws.cross_section_radius_m:
            errors.append("scene requires three non-interpenetrating stacked seals")
        if ws.segments < 32 or ws.segments % 2:
            errors.append("weatherstrip.segments must be an even integer >= 32")
        for name, value in (
            ("rest_radius_m", ws.rest_radius_m),
            ("minor_radius_m", ws.minor_radius_m),
            ("cross_section_radius_m", ws.cross_section_radius_m),
            ("total_mass_kg", ws.total_mass_kg),
            ("stretch_stiffness_n_m", ws.stretch_stiffness_n_m),
            ("shear_stiffness_n_m", ws.shear_stiffness_n_m),
            ("bend_stiffness_n_m", ws.bend_stiffness_n_m),
            ("twist_stiffness_n_m", ws.twist_stiffness_n_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                errors.append(f"weatherstrip.{name} must be finite and positive")
        for name in (
            "contact_stiffness_n_m",
            "contact_damping_n_s_m",
            "gripper_contact_stiffness_n_m",
            "gripper_contact_damping_n_s_m",
        ):
            value = getattr(sim, name)
            if not math.isfinite(value) or value <= 0:
                errors.append(f"simulation.{name} must be finite and positive")
        if (
            sim.gripper_contact_stiffness_n_m < 0.5 * sim.contact_stiffness_n_m
            or sim.gripper_contact_damping_n_s_m < 0.5 * sim.contact_damping_n_s_m
        ):
            errors.append("gripper contact pair coefficients would require negative shape coefficients")
        if ws.neighbor_chord_m <= 2.0 * ws.cross_section_radius_m:
            errors.append("capsule diameter must be smaller than the neighbor chord")
        if not 0.0 <= scene.table_friction <= 2.0:
            errors.append("scene.table_friction must be in [0, 2]")
        if not 0.0 <= ws.friction <= 2.0:
            errors.append("weatherstrip.friction must be in [0, 2]")
        if scene.robot_spacing_m <= 2.0 * (ws.rest_radius_m + traj.stretch_extension_m):
            errors.append("robot spacing must exceed the fully stretched grasp span")
        if scene.table_half_extents_m[0] <= ws.rest_radius_m + traj.stretch_extension_m:
            errors.append("table X half extent must contain the fully stretched ring")
        if scene.table_half_extents_m[1] <= ws.rest_radius_m:
            errors.append("table Y half extent must contain the rest ring")
        if not 0.0 <= assets.gripper_open_angle_rad < assets.gripper_hold_angle_rad <= math.radians(47):
            errors.append("Robotiq angles must satisfy 0 <= open < hold <= 47 degrees")
        if assets.gripper_grasp_offset_m <= 0.0:
            errors.append("gripper_grasp_offset_m must be positive")
        if sim.fps <= 0 or sim.substeps <= 0 or sim.solver_iterations <= 0:
            errors.append("simulation rates and VBD iteration count must be positive")
        if sim.ik_iterations <= 0 or sim.admm_iterations <= 0 or sim.mujoco_iterations <= 0:
            errors.append("IK, ADMM, and MuJoCo iteration counts must be positive")
        if any((not math.isfinite(value) or value <= 0.0) for value in traj.durations_s):
            errors.append("all trajectory phase durations must be finite and positive")
        if traj.approach_height_m <= 0.0 or traj.lift_height_m <= 0.0 or traj.stretch_extension_m <= 0.0:
            errors.append("trajectory heights and extension must be positive")
        if traj.clearance_m <= 0.0 or traj.sag_lowering_m < 0.0:
            errors.append("trajectory clearance must be positive and release lowering non-negative")
        if scene.table_half_extents_m[0] <= ws.rest_radius_m + traj.clearance_m:
            errors.append("table X half extent must contain the outward gripper clearance")
        if traj.grip_depth_m < 0.0:
            errors.append("trajectory.grip_depth_m must be non-negative")
        if require_asset:
            if not assets.robot_usd.is_file():
                errors.append(f"robot USD does not exist: {assets.robot_usd}")
            if not assets.gripper_usd.is_file():
                errors.append(f"gripper USD does not exist: {assets.gripper_usd}")
        if errors:
            raise ValueError("; ".join(errors))


def _resolve_asset(raw_path: str, environment_name: str) -> Path:
    override = os.environ.get(environment_name)
    selected = Path(override if override else raw_path).expanduser()
    if not selected.is_absolute():
        selected = (REPO_ROOT / selected).resolve()
    return selected


def load_config(path: str | Path = DEFAULT_CONFIG_PATH, *, require_asset: bool = True) -> ProjectConfig:
    source_path = Path(path).expanduser().resolve()
    with source_path.open("rb") as stream:
        raw = tomllib.load(stream)

    asset = raw["assets"]
    scene = raw["scene"]
    weatherstrip = raw["weatherstrip"]
    simulation = raw["simulation"]
    trajectory = raw["trajectory"]

    config = ProjectConfig(
        assets=AssetConfig(
            robot_usd=_resolve_asset(str(asset["robot_usd"]), "NEWTON_WEATHERSTRIP_ROBOT_USD"),
            robot_prim=str(asset["robot_prim"]),
            physics_variant=str(asset["physics_variant"]),
            simready_profile=str(asset["simready_profile"]),
            gripper_usd=_resolve_asset(str(asset["gripper_usd"]), "NEWTON_WEATHERSTRIP_GRIPPER_USD"),
            gripper_prim=str(asset["gripper_prim"]),
            gripper_physics_variant=str(asset["gripper_physics_variant"]),
            gripper_grasp_offset_m=float(asset["gripper_grasp_offset_m"]),
            gripper_open_angle_rad=float(asset["gripper_open_angle_rad"]),
            gripper_hold_angle_rad=float(asset["gripper_hold_angle_rad"]),
        ),
        scene=SceneConfig(
            robot_spacing_m=float(scene["robot_spacing_m"]),
            robot_base_z_m=float(scene["robot_base_z_m"]),
            table_half_extents_m=tuple(float(v) for v in scene["table_half_extents_m"]),
            table_center_z_m=float(scene["table_center_z_m"]),
            table_friction=float(scene["table_friction"]),
            stack_count=int(scene["stack_count"]),
            stack_pitch_m=float(scene["stack_pitch_m"]),
        ),
        weatherstrip=WeatherStripConfig(
            **{key: int(value) if key == "segments" else float(value) for key, value in weatherstrip.items()}
        ),
        simulation=SimulationConfig(
            fps=int(simulation["fps"]),
            substeps=int(simulation["substeps"]),
            solver_iterations=int(simulation["solver_iterations"]),
            ik_iterations=int(simulation["ik_iterations"]),
            admm_iterations=int(simulation["admm_iterations"]),
            mujoco_iterations=int(simulation["mujoco_iterations"]),
            gravity_m_s2=float(simulation["gravity_m_s2"]),
            contact_stiffness_n_m=float(simulation["contact_stiffness_n_m"]),
            contact_damping_n_s_m=float(simulation["contact_damping_n_s_m"]),
            gripper_contact_stiffness_n_m=float(simulation["gripper_contact_stiffness_n_m"]),
            gripper_contact_damping_n_s_m=float(simulation["gripper_contact_damping_n_s_m"]),
            gripper_drive_stiffness_nm_rad=float(simulation["gripper_drive_stiffness_nm_rad"]),
            gripper_drive_damping_nm_s_rad=float(simulation["gripper_drive_damping_nm_s_rad"]),
            gripper_max_torque_nm=float(simulation["gripper_max_torque_nm"]),
            device=str(simulation["device"]),
        ),
        trajectory=TrajectoryConfig(**{key: float(value) for key, value in trajectory.items()}),
        source_path=source_path,
    )
    config.validate(require_asset=require_asset)
    return config
