"""Deterministic dual-arm grip, stretch, lower, release, unload, outward-clear, and retract timeline."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ProjectConfig


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class PoseSample:
    phase: str
    left: Vec3
    right: Vec3
    grip_fraction: float
    progress: float

    @property
    def attached(self) -> bool:
        """Compatibility view: true when the physical gripper is mostly closed."""
        return self.grip_fraction >= 0.95


@dataclass(frozen=True)
class Phase:
    name: str
    duration_s: float
    left_end: Vec3
    right_end: Vec3
    grip_end: float

    @property
    def attached(self) -> bool:
        return self.grip_end >= 0.95


def _mix(a: Vec3, b: Vec3, alpha: float) -> Vec3:
    return tuple((1.0 - alpha) * x + alpha * y for x, y in zip(a, b, strict=True))


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


class DualArmTimeline:
    def __init__(self, config: ProjectConfig, grasp_centers=None):
        radius = config.weatherstrip.rest_radius_m
        extension = config.trajectory.stretch_extension_m
        grip_z = config.top_rest_height_m - config.trajectory.grip_depth_m
        approach_z = grip_z + config.trajectory.approach_height_m
        lift_z = grip_z + config.trajectory.lift_height_m
        drop_z = lift_z - config.trajectory.sag_lowering_m
        clear_x = radius + config.trajectory.clearance_m

        approach_left = (-radius, 0.0, approach_z)
        approach_right = (radius, 0.0, approach_z)
        grip_left = (-radius, 0.0, grip_z)
        grip_right = (radius, 0.0, grip_z)
        if grasp_centers is not None:
            grip_left = tuple(grasp_centers[0] - (0, 0, config.trajectory.grip_depth_m))
            grip_right = tuple(grasp_centers[1] - (0, 0, config.trajectory.grip_depth_m))
        lift_left = (grip_left[0], grip_left[1], lift_z)
        lift_right = (grip_right[0], grip_right[1], lift_z)
        place_left = (grip_left[0], grip_left[1], drop_z)
        place_right = (grip_right[0], grip_right[1], drop_z)
        stretch_left = (grip_left[0] - extension, grip_left[1], lift_z)
        stretch_right = (grip_right[0] + extension, grip_right[1], lift_z)
        retract_left = (-clear_x, 0.0, lift_z)
        retract_right = (clear_x, 0.0, lift_z)
        t = config.trajectory

        self.initial_left = approach_left
        self.initial_right = approach_right
        self.phases = (
            Phase("ready", t.ready_s, approach_left, approach_right, 0.0),
            Phase("descend", t.descend_s, grip_left, grip_right, 0.0),
            Phase("grasp", t.grasp_s, grip_left, grip_right, 1.0),
            Phase("lift", t.lift_s, lift_left, lift_right, 1.0),
            Phase("stretch", t.stretch_s, stretch_left, stretch_right, 1.0),
            Phase("hold", t.hold_s, stretch_left, stretch_right, 1.0),
            Phase("relax", t.relax_s, lift_left, lift_right, 1.0),
            Phase("sag", t.sag_s, place_left, place_right, 1.0),
            Phase("release", t.release_s, place_left, place_right, 0.0),
            Phase("fall", t.fall_s, place_left, place_right, 0.0),
            Phase("retract", t.retract_s, retract_left, retract_right, 0.0),
            Phase("settle", t.settle_s, retract_left, retract_right, 0.0),
        )
        self.total_duration_s = sum(phase.duration_s for phase in self.phases)

    def sample(self, time_s: float) -> PoseSample:
        elapsed = max(0.0, time_s)
        left_start = self.initial_left
        right_start = self.initial_right
        grip_start = 0.0

        for phase in self.phases:
            if elapsed <= phase.duration_s:
                raw = elapsed / phase.duration_s
                alpha = _smoothstep(raw)
                return PoseSample(
                    phase=phase.name,
                    left=_mix(left_start, phase.left_end, alpha),
                    right=_mix(right_start, phase.right_end, alpha),
                    grip_fraction=(1.0 - alpha) * grip_start + alpha * phase.grip_end,
                    progress=raw,
                )
            elapsed -= phase.duration_s
            left_start = phase.left_end
            right_start = phase.right_end
            grip_start = phase.grip_end

        last = self.phases[-1]
        return PoseSample(last.name, last.left_end, last.right_end, last.grip_end, 1.0)
