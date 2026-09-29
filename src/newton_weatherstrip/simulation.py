"""Two FANUC arms with Robotiq 2F-85 EEFs and a closed Newton VBD door seal."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM

from .config import DEFAULT_CONFIG_PATH, REPO_ROOT, ProjectConfig, load_config
from .asset import load_weatherstrip_points
from .trajectory import DualArmTimeline, PoseSample


FANUC_SEED_Q = [0.0, 0.7, -1.4, 0.0, -0.9, 0.0]
# Official FANUC J6 origin is 160 mm behind the +X flange face.
# Gripper local +Z is mounted along flange +X.
GRIPPER_MOUNT = wp.transform(wp.vec3(0.160, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2))
GRIPPER_DOWN_XYZW = tuple(
    wp.mul(wp.quat(2.0**-0.5, 2.0**-0.5, 0.0, 0.0), wp.quat_inverse(wp.transform_get_rotation(GRIPPER_MOUNT)))
)


@wp.kernel
def _scatter_control_targets(
    ik_q: wp.array2d[float],
    joint_target_q: wp.array[float],
    left_arm_start: int,
    right_arm_start: int,
    left_gripper_coord: int,
    right_gripper_coord: int,
    gripper_target: float,
):
    index = wp.tid()
    if index < 6:
        joint_target_q[left_arm_start + index] = ik_q[0, index]
    else:
        joint_target_q[right_arm_start + index - 6] = ik_q[0, index]
    if index == 0:
        joint_target_q[left_gripper_coord] = gripper_target
        joint_target_q[right_gripper_coord] = gripper_target


@wp.kernel
def _contact_summary(
    count: wp.array(dtype=int),
    first: wp.array(dtype=int),
    second: wp.array(dtype=int),
    strip_of_shape: wp.array(dtype=int),
    finger_side: wp.array(dtype=int),
    table_shape: int,
    counts: wp.array2d(dtype=int),
):
    i = wp.tid()
    if i < count[0]:
        a, b = first[i], second[i]
        if a >= 0 and b >= 0:
            sa, sb = strip_of_shape[a], strip_of_shape[b]
            if sa >= 0:
                if finger_side[b] >= 0:
                    wp.atomic_add(counts, sa, finger_side[b], 1)
                if b == table_shape:
                    wp.atomic_add(counts, sa, 2, 1)
                if sb >= 0 and sb != sa:
                    wp.atomic_add(counts, sa, 3 + sb, 1)
            if sb >= 0:
                if finger_side[a] >= 0:
                    wp.atomic_add(counts, sb, finger_side[a], 1)
                if a == table_shape:
                    wp.atomic_add(counts, sb, 2, 1)
                if sa >= 0 and sa != sb:
                    wp.atomic_add(counts, sb, 3 + sa, 1)


@wp.kernel
def _record_contact_peak(count: wp.array(dtype=int), peak: wp.array(dtype=int)):
    wp.atomic_max(peak, 0, count[0])


def _find_suffix(mapping: dict[str, int], suffix: str) -> int:
    matches = [index for path, index in mapping.items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one path ending in {suffix!r}, got {matches}")
    return matches[0]


def _open_variant_stage(path: Path, variant: str):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"USD has no default prim: {path}")
    physics = default_prim.GetVariantSets().GetVariantSet("Physics")
    names = list(physics.GetVariantNames())
    if variant not in names:
        raise RuntimeError(f"Physics variant {variant!r} is not available in {path}: {names}")
    if not physics.SetVariantSelection(variant):
        raise RuntimeError(f"Could not select Physics={variant!r} in {path}")
    return stage


def _transform_point(transform_values: np.ndarray, local: tuple[float, float, float]) -> np.ndarray:
    transform = wp.transform(
        wp.vec3(float(transform_values[0]), float(transform_values[1]), float(transform_values[2])),
        wp.quat(
            float(transform_values[3]),
            float(transform_values[4]),
            float(transform_values[5]),
            float(transform_values[6]),
        ),
    )
    return np.asarray(wp.transform_point(transform, wp.vec3(*local)), dtype=np.float32)


class WeatherStripSimulation:
    def __init__(self, viewer, args: argparse.Namespace, config: ProjectConfig):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.args = args
        self.config = config
        self.timeline = DualArmTimeline(config)
        self.frame_dt = 1.0 / config.simulation.fps
        self.sim_substeps = int(args.substeps or config.simulation.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.vbd_iterations = int(args.solver_iterations or config.simulation.solver_iterations)
        self.ik_iterations = int(args.ik_iterations or config.simulation.ik_iterations)
        self.admm_iterations = int(args.admm_iterations or config.simulation.admm_iterations)
        self.mujoco_iterations = int(args.mujoco_iterations or config.simulation.mujoco_iterations)
        self.physics_graph = None
        self.ik_graph = None
        self.use_cuda_graph = (
            os.environ.get("WEATHERSTRIP_CUDA_GRAPH", os.environ.get("EFORREST_CUDA_GRAPH", "0")) == "1"
        )
        self.sim_time = 0.0
        self.last_sample = self.timeline.sample(0.0)
        self.pick_centers = None

        self.max_weatherstrip_span_m = 0.0
        self.max_tool_target_error_m = 0.0
        self.minimum_weatherstrip_surface_z_m = math.inf
        self.max_grip_distance_m = 0.0
        self.max_observed_jaw_angle_rad = 0.0
        self.left_grip_contact_frames = 0
        self.right_grip_contact_frames = 0
        self.saw_commanded_grip = False
        self.saw_commanded_release = False
        self.saw_jaw_close = False
        self.saw_jaw_reopen = False

        self._build_scene()
        self._build_runtime()
        self._configure_viewer()
        self.phase_metrics = []
        centers = self.state_0.body_q.numpy()[self.weatherstrip_bodies, :3]
        self.rest_centerline_length = float(np.linalg.norm(np.roll(centers, -1, axis=0) - centers, axis=1).sum())
        self.max_centerline_strain = 0.0
        self.max_lift_height = 0.0
        self.lower_max_lift_m = np.zeros(len(self.strips) - 1)
        self.lower_gripper_contact_frames = 0
        self.release_centroid_z = None
        self.release_clearance_m = None
        self.freefall_observed = False
        self.drop_impact_frames = 0
        self.drop_trace = []
        self.motion_trace = []
        self.previous_strip_centers = None
        self.presentation = None
        if isinstance(viewer, newton.viewer.ViewerUSD):
            from .presentation import Presentation

            viewer.begin_frame(0.0)
            viewer.log_state(self.state_0)
            self.presentation = Presentation(viewer, self)

    def _add_robot(
        self,
        builder: newton.ModelBuilder,
        robot_stage,
        x_m: float,
        yaw_rad: float,
    ) -> tuple[int, int]:
        coord_start = builder.joint_coord_count
        result = builder.add_usd(
            robot_stage,
            xform=wp.transform(
                wp.vec3(x_m, 0.0, self.config.scene.robot_base_z_m),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw_rad),
            ),
            floating=False,
            override_root_xform=True,
            root_path=self.config.assets.robot_prim,
            enable_self_collisions=False,
            load_static_visual_shapes=False,
            hide_collision_shapes=True,
        )
        ee_body = _find_suffix(result["path_body_map"], "/J6_link")
        if builder.joint_coord_count - coord_start != 6:
            raise RuntimeError("FANUC import must contribute exactly six coordinates")
        builder.joint_q[coord_start : coord_start + 6] = FANUC_SEED_Q
        builder.joint_target_q[coord_start : coord_start + 6] = FANUC_SEED_Q
        return ee_body, coord_start

    def _add_assembly(self, builder, robot_stage, gripper_stage, x_m, yaw_rad):
        body_start, joint_start, shape_start = builder.body_count, builder.joint_count, builder.shape_count
        ee_body, arm_coord_start = self._add_robot(builder, robot_stage, x_m, yaw_rad)
        gripper_body_start, gripper_shape_start = builder.body_count, builder.shape_count
        gripper_joint_start = builder.joint_count
        result = builder.add_usd(
            gripper_stage,
            xform=GRIPPER_MOUNT,
            floating=False,
            parent_body=ee_body,
            override_root_xform=True,
            root_path=self.config.assets.gripper_prim,
            enable_self_collisions=False,
            load_static_visual_shapes=False,
            hide_collision_shapes=True,
        )
        master_joint = next(
            j
            for j in range(gripper_joint_start, builder.joint_count)
            if str(builder.joint_label[j]).endswith("/Joints/finger_joint")
        )
        gripper_coord = int(builder.joint_q_start[master_joint])
        finger_bodies = {i for p, i in result["path_body_map"].items() if "finger" in p or "knuckle" in p}
        for shape in range(gripper_shape_start, builder.shape_count):
            label = str(builder.shape_label[shape])
            builder.shape_color[shape] = (
                (0.018, 0.020, 0.022)
                if "fingertips" in label
                else ((0.40, 0.44, 0.48) if "knuckle" in label or "finger" in label else (0.08, 0.10, 0.13))
            )
        return dict(
            body_start=body_start,
            body_end=builder.body_count,
            joint_start=joint_start,
            joint_end=builder.joint_count,
            shape_start=shape_start,
            shape_end=builder.shape_count,
            ee_body=ee_body,
            arm_coord_start=arm_coord_start,
            gripper_coord=gripper_coord,
            gripper_bodies=list(range(gripper_body_start, builder.body_count)),
            gripper_shapes=list(range(gripper_shape_start, builder.shape_count)),
            finger_shapes=[
                i for i in range(gripper_shape_start, builder.shape_count) if builder.shape_body[i] in finger_bodies
            ],
        )

    def _configure_drives(self, builder: newton.ModelBuilder, assemblies: list[dict[str, object]]) -> None:
        sim = self.config.simulation
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}

        for assembly in assemblies:
            arm_start = int(assembly["arm_coord_start"])
            arm_slice = slice(arm_start, arm_start + 6)
            builder.joint_target_ke[arm_slice] = [1800.0] * 6
            builder.joint_target_kd[arm_slice] = [130.0] * 6
            builder.joint_effort_limit[arm_slice] = [300.0] * 6
            builder.joint_armature[arm_slice] = [0.05] * 6
            builder.joint_target_mode[arm_slice] = [int(newton.JointTargetMode.POSITION_VELOCITY)] * 6

            for gripper_coord in (int(assembly["gripper_coord"]),):
                builder.joint_q[gripper_coord] = self.config.assets.gripper_open_angle_rad
                builder.joint_target_q[gripper_coord] = self.config.assets.gripper_open_angle_rad
                builder.joint_target_ke[gripper_coord] = sim.gripper_drive_stiffness_nm_rad
                builder.joint_target_kd[gripper_coord] = sim.gripper_drive_damping_nm_s_rad
                builder.joint_effort_limit[gripper_coord] = sim.gripper_max_torque_nm
                builder.joint_target_mode[gripper_coord] = int(newton.JointTargetMode.POSITION_VELOCITY)

            for body in range(int(assembly["body_start"]), int(assembly["body_end"])):
                gravcomp.values[body] = 1.0

    def _build_scene(self) -> None:
        cfg = self.config
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, cfg.simulation.gravity_m_s2))
        builder.rigid_gap = 0.003
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder)

        robot_stage = _open_variant_stage(cfg.assets.robot_usd, cfg.assets.physics_variant)
        gripper_stage = _open_variant_stage(cfg.assets.gripper_usd, cfg.assets.gripper_physics_variant)
        half_spacing = 0.5 * cfg.scene.robot_spacing_m
        assemblies = [
            self._add_assembly(builder, robot_stage, gripper_stage, -half_spacing, 0.0),
            self._add_assembly(builder, robot_stage, gripper_stage, half_spacing, math.pi),
        ]
        if builder.joint_coord_count != 24:
            raise RuntimeError(
                "Unexpected two-assembly topology before weatherstrip: "
                f"{builder.body_count} bodies, {builder.joint_count} joints, "
                f"{builder.joint_coord_count} coordinates"
            )
        self._configure_drives(builder, assemblies)

        self.assemblies = assemblies
        self.robot_gripper_bodies = [
            body for assembly in assemblies for body in range(int(assembly["body_start"]), int(assembly["body_end"]))
        ]
        self.robot_gripper_joints = [
            joint
            for assembly in assemblies
            for joint in range(int(assembly["joint_start"]), int(assembly["joint_end"]))
        ]
        self.arm_coord_starts = [int(assembly["arm_coord_start"]) for assembly in assemblies]
        self.gripper_coords = [int(assembly["gripper_coord"]) for assembly in assemblies]
        self.ee_body_indices = [int(assembly["ee_body"]) for assembly in assemblies]
        self.gripper_shape_sets = [set(int(shape) for shape in assembly["finger_shapes"]) for assembly in assemblies]

        table_hx, table_hy, table_hz = cfg.scene.table_half_extents_m
        surface_cfg = newton.ModelBuilder.ShapeConfig(
            ke=cfg.simulation.contact_stiffness_n_m,
            kd=cfg.simulation.contact_damping_n_s_m,
            mu=cfg.scene.table_friction,
            margin=0.001,
            gap=0.002,
        )
        self.table_shape = builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, cfg.scene.table_center_z_m),
                wp.quat_identity(),
            ),
            hx=table_hx,
            hy=table_hy,
            hz=table_hz,
            cfg=surface_cfg,
            label="weatherstrip_work_table",
            color=(0.22, 0.25, 0.29),
        )
        self.ground_shape = builder.add_ground_plane(height=0.0, cfg=surface_cfg, label="robot_floor")

        weatherstrip_cfg = newton.ModelBuilder.ShapeConfig(
            density=cfg.weatherstrip.rod_density_kg_m3,
            ke=cfg.simulation.contact_stiffness_n_m,
            kd=cfg.simulation.contact_damping_n_s_m,
            mu=cfg.weatherstrip.friction,
            margin=0.001,
            gap=0.002,
        )
        self.strips = []
        rest_points = np.asarray(load_weatherstrip_points(cfg))
        for layer in range(cfg.scene.stack_count):
            # Crossing support points keep round EPDM sections from rolling off
            # a perfectly collinear circular-section stack. All seals remain dynamic.
            angle = math.radians((30.0, -30.0, 0.0)[layer])
            rotation = np.array(
                [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
            )
            points = rest_points @ rotation.T + np.array([0, 0, layer * cfg.scene.stack_pitch_m])
            shape_start = builder.shape_count
            bodies, joints = builder.add_rod(
                positions=points.tolist(),
                radius=cfg.weatherstrip.cross_section_radius_m,
                body_frame_origin="com",
                cfg=weatherstrip_cfg,
                stretch_stiffness=cfg.weatherstrip.stretch_stiffness_n_m,
                stretch_damping=cfg.weatherstrip.stretch_damping_n_s_m,
                shear_stiffness=cfg.weatherstrip.shear_stiffness_n_m,
                shear_damping=cfg.weatherstrip.shear_damping_n_s_m,
                bend_stiffness=cfg.weatherstrip.bend_stiffness_n_m,
                bend_damping=cfg.weatherstrip.bend_damping_n_m_s,
                twist_stiffness=cfg.weatherstrip.twist_stiffness_n_m,
                twist_damping=cfg.weatherstrip.twist_damping_n_m_s,
                closed=True,
                label=f"weatherstrip_{layer}",
                color=(0.004, 0.004, 0.004),
            )
            shapes = list(range(shape_start, builder.shape_count))
            assert len(bodies) == len(joints) == cfg.weatherstrip.segments
            self.strips.append(dict(bodies=bodies, joints=joints, shapes=shapes))
            # Only neighbors of the SAME continuous seal are excluded.
            for i in range(len(shapes)):
                for offset in (1, 2):
                    x, y = shapes[i], shapes[(i + offset) % len(shapes)]
                    builder.shape_collision_filter_pairs.append((min(x, y), max(x, y)))
            if isinstance(self.viewer, newton.viewer.ViewerUSD):
                for shape in shapes:
                    builder.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
        top = self.strips[-1]
        self.weatherstrip_bodies, self.weatherstrip_joints, self.weatherstrip_shapes = (
            top["bodies"],
            top["joints"],
            top["shapes"],
        )
        self.all_strip_bodies = [b for strip in self.strips for b in strip["bodies"]]
        self.all_strip_joints = [j for strip in self.strips for j in strip["joints"]]
        self.all_strip_shapes = [sh for strip in self.strips for sh in strip["shapes"]]

        builder.color()
        self.model = builder.finalize(device=self.args.device)
        self.device = self.model.device
        qstarts = self.model.joint_q_start.numpy()
        self.mimic_follower_coords = qstarts[self.model.constraint_mimic_joint0.numpy()]
        self.mimic_leader_coords = qstarts[self.model.constraint_mimic_joint1.numpy()]
        self.mimic_coefficients = self.model.constraint_mimic_coef1.numpy()
        self.max_mimic_error_rad = 0.0
        # Author nominal pair coefficients using VBD's arithmetic material mixing.
        # Seal/seal and seal/table contacts are resolved by VBD. Cross-entry
        # fingertip/seal contacts belong to ADMM's private contact stream; these
        # shape gains must not be interpreted as its interface penalty rho.
        material_ke = np.full(self.model.shape_count, cfg.simulation.contact_stiffness_n_m, dtype=np.float32)
        material_kd = np.full(self.model.shape_count, cfg.simulation.contact_damping_n_s_m, dtype=np.float32)
        for shapes in self.gripper_shape_sets:
            material_ke[list(shapes)] = (
                2 * cfg.simulation.gripper_contact_stiffness_n_m - cfg.simulation.contact_stiffness_n_m
            )
            material_kd[list(shapes)] = (
                2 * cfg.simulation.gripper_contact_damping_n_s_m - cfg.simulation.contact_damping_n_s_m
            )
        self.model.shape_material_ke.assign(material_ke)
        self.model.shape_material_kd.assign(material_kd)
        self.model.shape_material_mu.fill_(cfg.weatherstrip.friction)
        material_mu = np.full(len(self.model.shape_material_mu), cfg.weatherstrip.friction, dtype=np.float32)
        material_mu[[self.table_shape, self.ground_shape]] = cfg.scene.table_friction
        self.model.shape_material_mu.assign(material_mu)
        self.weatherstrip_shape_set = set(self.weatherstrip_shapes)
        self.admm_gripper_weatherstrip_pair_count = self._count_candidate_contact_pairs()
        if self.admm_gripper_weatherstrip_pair_count <= 0:
            raise RuntimeError("No gripper-weatherstrip collision pairs were generated")

    def _count_candidate_contact_pairs(self) -> int:
        gripper_shapes = set().union(*self.gripper_shape_sets)
        count = 0
        for first, second in self.model.shape_contact_pairs.numpy():
            pair = {int(first), int(second)}
            if pair & gripper_shapes and pair & self.weatherstrip_shape_set:
                count += 1
        return count

    def _build_ik(self) -> None:
        cfg = self.config
        robot_stage = _open_variant_stage(cfg.assets.robot_usd, cfg.assets.physics_variant)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, cfg.simulation.gravity_m_s2))
        half_spacing = 0.5 * cfg.scene.robot_spacing_m
        left_ee, left_start = self._add_robot(builder, robot_stage, -half_spacing, 0.0)
        right_ee, right_start = self._add_robot(builder, robot_stage, half_spacing, math.pi)
        if (left_start, right_start, builder.joint_coord_count) != (0, 6, 12):
            raise RuntimeError("Unexpected dual-FANUC IK coordinate layout")
        self.ik_model = builder.finalize(device=self.device)

        initial = self.timeline.sample(0.0)
        self.ik_position_targets = [
            wp.array([initial.left], dtype=wp.vec3, device=self.device),
            wp.array([initial.right], dtype=wp.vec3, device=self.device),
        ]
        self.ik_rotation_targets = [
            wp.array([GRIPPER_DOWN_XYZW], dtype=wp.vec4, device=self.device),
            wp.array([GRIPPER_DOWN_XYZW], dtype=wp.vec4, device=self.device),
        ]
        objectives: list[ik.IKObjective] = []
        for ee_body, position_target, rotation_target in zip(
            (left_ee, right_ee),
            self.ik_position_targets,
            self.ik_rotation_targets,
            strict=True,
        ):
            objectives.append(
                ik.IKObjectivePosition(
                    link_index=ee_body,
                    link_offset=wp.vec3(0.160 + cfg.assets.gripper_grasp_offset_m, 0.0, 0.0),
                    target_positions=position_target,
                    weight=1.0,
                )
            )
            objectives.append(
                ik.IKObjectiveRotation(
                    link_index=ee_body,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=rotation_target,
                    weight=1.0,
                )
            )
        objectives.append(
            ik.IKObjectiveJointLimit(
                joint_limit_lower=self.ik_model.joint_limit_lower,
                joint_limit_upper=self.ik_model.joint_limit_upper,
                weight=2.0,
            )
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=objectives,
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_joint_q = wp.array(FANUC_SEED_Q * 2, dtype=wp.float32, device=self.device).reshape((1, 12))
        self.ik_solver.step(
            self.ik_joint_q,
            self.ik_joint_q,
            iterations=max(80, self.ik_iterations),
        )

    def _scatter_targets(self, target_array: wp.array, gripper_target: float) -> None:
        wp.launch(
            _scatter_control_targets,
            dim=12,
            inputs=[
                self.ik_joint_q,
                target_array,
                self.arm_coord_starts[0],
                self.arm_coord_starts[1],
                self.gripper_coords[0],
                self.gripper_coords[1],
                gripper_target,
            ],
            device=self.device,
        )

    def _make_mujoco_solver(self, view):
        solver = SolverMuJoCo(
            model=view,
            solver="cg",
            integrator="implicitfast",
            iterations=self.mujoco_iterations,
            ls_iterations=4,
            use_mujoco_contacts=False,
            njmax=128,
            nconmax=32,
        )
        # The closed finger linkage must resist contact loads. MuJoCo's default
        # soft equality permits the passive finger joints to open under load.
        assert solver.mj_model.neq == 10, "Expected five linkage equalities per 2F-85"
        solver.mj_model.eq_solref[:] = [0.004, 1.0]
        solver.mj_model.eq_solimp[:] = [0.99, 0.99, 0.001, 0.5, 2.0]
        solver.mjw_model.eq_solref.assign(solver.mj_model.eq_solref[None].astype(np.float32))
        solver.mjw_model.eq_solimp.assign(solver.mj_model.eq_solimp[None].astype(np.float32))
        print(f"MJC buffers: neq={solver.mj_model.neq} njmax={solver.mjw_data.njmax}", flush=True)
        return solver

    def _build_runtime(self) -> None:
        self._build_ik()
        open_angle = self.config.assets.gripper_open_angle_rad
        self._scatter_targets(self.model.joint_q, open_angle)
        self._scatter_targets(self.model.joint_target_q, open_angle)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.control = self.model.control()
        self._scatter_targets(self.control.joint_target_q, open_angle)

        self.solver = SolverCoupledADMM(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=self._make_mujoco_solver,
                    bodies=self.robot_gripper_bodies,
                    joints=self.robot_gripper_joints,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda view: SolverVBD(
                        model=view,
                        iterations=self.vbd_iterations,
                        rigid_contact_history=False,
                        rigid_contact_hard=True,
                        friction_epsilon=0.01,
                    ),
                    bodies=self.all_strip_bodies,
                    joints=self.all_strip_joints,
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=self.admm_iterations,
                rho=200.0,
                gamma=0.001,
                baumgarte=0.5,
                rigid_contact_matching="latest",
                contact_matching_force_scale=0.9,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(source="mjc", destination="vbd"),
                ],
            ),
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=8192)
        self.contacts = self.collision_pipeline.contacts()
        self.contact_peak = wp.zeros(1, dtype=int, device=self.device)
        print(f"Collision capacity: {len(self.contacts.rigid_contact_shape0)}", flush=True)
        self.solver.prepare_contacts(self.contacts)
        strip_of_shape = np.full(self.model.shape_count, -1, dtype=np.int32)
        finger_side = np.full(self.model.shape_count, -1, dtype=np.int32)
        for i, strip in enumerate(self.strips):
            strip_of_shape[strip["shapes"]] = i
        for i, shapes in enumerate(self.gripper_shape_sets):
            finger_side[list(shapes)] = i
        self.strip_of_shape = wp.array(strip_of_shape, dtype=int, device=self.device)
        self.finger_side = wp.array(finger_side, dtype=int, device=self.device)
        self.contact_counts = wp.zeros((len(self.strips), 3 + len(self.strips)), dtype=int, device=self.device)
        self.contact_counts_host = self.contact_counts.numpy()

    def _configure_viewer(self) -> None:
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(pos=wp.vec3(1.55, -1.85, 1.35), pitch=-23.0, yaw=139.0)
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.50))

    def _update_targets(self, sample: PoseSample) -> float:
        self.ik_position_targets[0].fill_(wp.vec3(*sample.left))
        self.ik_position_targets[1].fill_(wp.vec3(*sample.right))
        if self.use_cuda_graph and self.device.is_cuda:
            if self.ik_graph is None:
                with wp.ScopedCapture(device=self.device) as cap:
                    self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iterations)
                self.ik_graph = cap.graph
            wp.capture_launch(self.ik_graph)
        else:
            self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iterations)

        assets = self.config.assets
        jaw_target = (
            1.0 - sample.grip_fraction
        ) * assets.gripper_open_angle_rad + sample.grip_fraction * assets.gripper_hold_angle_rad
        self._scatter_targets(self.control.joint_target_q, jaw_target)
        if sample.grip_fraction >= 0.95:
            self.saw_commanded_grip = True
        if self.saw_commanded_grip and sample.grip_fraction <= 0.05:
            self.saw_commanded_release = True
        return jaw_target

    def _tool_positions(self, body_q: np.ndarray) -> np.ndarray:
        offset = (0.160 + self.config.assets.gripper_grasp_offset_m, 0.0, 0.0)
        return np.stack([_transform_point(body_q[index], offset) for index in self.ee_body_indices])

    def _collect_contacts(self, sample):
        self.contact_counts.zero_()
        wp.launch(
            _contact_summary,
            dim=len(self.contacts.rigid_contact_shape0),
            inputs=[
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.strip_of_shape,
                self.finger_side,
                self.table_shape,
                self.contact_counts,
            ],
            device=self.device,
        )
        self.contact_counts_host = self.contact_counts.numpy()
        if sample.grip_fraction >= 0.5:
            self.left_grip_contact_frames += int(self.contact_counts_host[-1, 0] > 0)
            self.right_grip_contact_frames += int(self.contact_counts_host[-1, 1] > 0)

    def _collect_metrics(self, sample: PoseSample) -> None:
        body_q = self.state_0.body_q.numpy()
        self.body_q_host = body_q
        weatherstrip_q = body_q[self.weatherstrip_bodies, :3]
        span = float(np.ptp(weatherstrip_q[:, :2], axis=0).max())
        self.max_weatherstrip_span_m = max(self.max_weatherstrip_span_m, span)
        length = float(np.linalg.norm(np.roll(weatherstrip_q, -1, axis=0) - weatherstrip_q, axis=1).sum())
        self.max_centerline_strain = max(self.max_centerline_strain, length / self.rest_centerline_length - 1.0)
        self.max_lift_height = max(
            self.max_lift_height, float(weatherstrip_q[:, 2].min()) - self.config.top_rest_height_m
        )
        if not self.phase_metrics or self.phase_metrics[-1]["phase"] != sample.phase:
            self.phase_metrics.append(
                {
                    "phase": sample.phase,
                    "time_s": self.sim_time,
                    "span_m": span,
                    "minimum_center_z_m": float(weatherstrip_q[:, 2].min()),
                    "jaw_angles_rad": self.state_0.joint_q.numpy()[self.gripper_coords].tolist(),
                }
            )
            print(f"PHASE {sample.phase}: t={self.sim_time:.2f}s span={span:.4f}m", flush=True)
        self.minimum_weatherstrip_surface_z_m = min(
            self.minimum_weatherstrip_surface_z_m,
            float(weatherstrip_q[:, 2].min() - self.config.weatherstrip.cross_section_radius_m),
        )

        tools = self._tool_positions(body_q)
        targets = np.asarray([sample.left, sample.right], dtype=np.float32)
        self.max_tool_target_error_m = max(
            self.max_tool_target_error_m,
            float(np.linalg.norm(tools - targets, axis=1).max()),
        )
        if sample.grip_fraction >= 0.95 and sample.phase not in {"grasp", "release"}:
            nearest = [float(np.linalg.norm(weatherstrip_q - tool, axis=1).min()) for tool in tools]
            self.max_grip_distance_m = max(self.max_grip_distance_m, max(nearest))

        joint_q = self.state_0.joint_q.numpy()
        jaw_angles = joint_q[self.gripper_coords]
        mimic_error = np.max(
            np.abs(joint_q[self.mimic_follower_coords] - self.mimic_coefficients * joint_q[self.mimic_leader_coords])
        )
        self.max_mimic_error_rad = max(self.max_mimic_error_rad, float(mimic_error))
        observed_jaw = float(np.max(jaw_angles))
        self.max_observed_jaw_angle_rad = max(self.max_observed_jaw_angle_rad, observed_jaw)
        if float(np.min(jaw_angles)) >= 0.40:
            self.saw_jaw_close = True
        if self.saw_jaw_close and sample.grip_fraction <= 0.05 and float(np.max(np.abs(jaw_angles))) <= 0.12:
            self.saw_jaw_reopen = True
        self._collect_contacts(sample)
        layers = [body_q[strip["bodies"], :3] for strip in self.strips]
        stacked = np.asarray(layers)
        if self.previous_strip_centers is not None:
            speeds = np.linalg.norm(stacked - self.previous_strip_centers, axis=2) / self.frame_dt
            self.motion_trace.append(
                dict(
                    time_s=self.sim_time,
                    phase=sample.phase,
                    rms_speed_m_s=np.sqrt(np.mean(speeds**2, axis=1)).tolist(),
                    max_speed_m_s=speeds.max(axis=1).tolist(),
                )
            )
        self.previous_strip_centers = stacked.copy()
        if self.release_centroid_z is None:
            for i, points in enumerate(layers[:-1]):
                self.lower_max_lift_m[i] = max(
                    self.lower_max_lift_m[i],
                    float(points[:, 2].mean()) - (self.config.rest_height_m + i * self.config.scene.stack_pitch_m),
                )
            if sample.phase in {"grasp", "lift", "stretch", "hold", "relax", "sag"} and np.any(
                self.contact_counts_host[:-1, :2]
            ):
                self.lower_gripper_contact_frames += 1
        if sample.phase == "release" and self.release_centroid_z is None:
            self.release_centroid_z = float(weatherstrip_q[:, 2].mean())
            self.release_clearance_m = float(
                weatherstrip_q[:, 2].min()
                - max(p[:, 2].max() for p in layers[:-1])
                - 2 * self.config.weatherstrip.cross_section_radius_m
            )
        if sample.phase in {"release", "fall", "retract", "settle"}:
            grip_contacts = int(self.contact_counts_host[-1, :2].sum())
            lower_contacts = int(self.contact_counts_host[-1, 3:-1].sum())
            clearance = float(
                weatherstrip_q[:, 2].min()
                - max(p[:, 2].max() for p in layers[:-1])
                - 2 * self.config.weatherstrip.cross_section_radius_m
            )
            if (
                sample.grip_fraction < 0.9
                and grip_contacts == 0
                and lower_contacts == 0
                and self.contact_counts_host[-1, 2] == 0
                and clearance > 0.02
            ):
                self.freefall_observed = True
            if self.freefall_observed and lower_contacts > 0:
                self.drop_impact_frames += 1
            if int(round(self.sim_time / self.frame_dt)) % 6 == 0:
                self.drop_trace.append(
                    dict(
                        time_s=self.sim_time,
                        centroid_z_m=float(weatherstrip_q[:, 2].mean()),
                        lower_contacts=lower_contacts,
                        gripper_contacts=grip_contacts,
                        clearance_m=clearance,
                    )
                )

    def step(self) -> None:
        if (
            self.config.trajectory.ready_s
            <= self.sim_time
            < self.config.trajectory.ready_s + self.config.trajectory.descend_s + 0.7 * self.config.trajectory.grasp_s
        ):
            points = self.state_0.body_q.numpy()[self.weatherstrip_bodies, :3]
            order = np.argsort(points[:, 0])
            self.pick_centers = np.stack((points[order[:2]].mean(axis=0), points[order[-2:]].mean(axis=0)))
            self.timeline = DualArmTimeline(self.config, self.pick_centers)
            if self.last_sample.phase == "ready":
                print(f"Sensed top grasp centers: {self.pick_centers.tolist()}", flush=True)
        sample = self.timeline.sample(self.sim_time)
        self.last_sample = sample
        self._update_targets(sample)

        if self.use_cuda_graph and self.device.is_cuda and self.sim_time > 0:
            if self.physics_graph is None:
                with wp.ScopedCapture(device=self.device) as capture:
                    self._physics_substeps()
                self.physics_graph = capture.graph
            wp.capture_launch(self.physics_graph)
        else:
            self._physics_substeps()

        self._collect_metrics(sample)
        self.sim_time += self.frame_dt

    def _physics_substeps(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            wp.launch(
                _record_contact_peak,
                dim=1,
                inputs=[self.contacts.rigid_contact_count, self.contact_peak],
                device=self.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.body_q_host = getattr(self, "body_q_host", None)
        self.viewer.log_state(self.state_0)
        if self.presentation is not None:
            self.presentation.update()
        self.viewer.end_frame()

    def _metrics(self) -> dict[str, object]:
        body_q = self.state_0.body_q.numpy()
        self.body_q_host = body_q
        weatherstrip_q = body_q[self.weatherstrip_bodies, :3]
        weatherstrip_extent = np.ptp(weatherstrip_q, axis=0)
        table_support_ceiling = self.config.scene.table_top_z_m + 0.025
        table_supported_fraction = float(
            np.mean(
                (weatherstrip_q[:, 2] <= table_support_ceiling)
                & (weatherstrip_q[:, 2] >= self.config.scene.table_top_z_m - 0.005)
                & (np.abs(weatherstrip_q[:, 0]) <= self.config.scene.table_half_extents_m[0])
                & (np.abs(weatherstrip_q[:, 1]) <= self.config.scene.table_half_extents_m[1])
            )
        )
        joint_q = self.state_0.joint_q.numpy()
        full_cycle_observed = self.sim_time + 0.5 * self.frame_dt >= self.timeline.total_duration_s
        contact_labels = {}
        ncontacts = int(self.contacts.rigid_contact_count.numpy()[0])
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:ncontacts]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:ncontacts]
        for a, b in zip(shape0, shape1):
            a, b = int(a), int(b)
            other = b if a in self.weatherstrip_shape_set else (a if b in self.weatherstrip_shape_set else -1)
            if other >= 0 and other not in self.weatherstrip_shape_set:
                label = self.model.shape_label[other]
                contact_labels[label] = contact_labels.get(label, 0) + 1
        tail = [sample for sample in self.motion_trace if sample["time_s"] >= self.sim_time - 1.0]
        tail_rms = np.mean([sample["rms_speed_m_s"] for sample in tail], axis=0).tolist() if tail else []
        tail_max = np.max([sample["max_speed_m_s"] for sample in tail], axis=0).tolist() if tail else []
        return {
            "schema_version": 6,
            "configuration": {
                "weatherstrip": asdict(self.config.weatherstrip),
                "scene": asdict(self.config.scene),
                "simulation": asdict(self.config.simulation),
                "trajectory": asdict(self.config.trajectory),
            },
            "final_second_mean_rms_speed_m_s": tail_rms,
            "final_second_max_node_speed_m_s": tail_max,
            "peak_rigid_contacts": int(self.contact_peak.numpy()[0]),
            "rigid_contact_capacity": len(self.contacts.rigid_contact_shape0),
            "sensed_top_grasp_centers_m": None if self.pick_centers is None else self.pick_centers.tolist(),
            "stack_count": len(self.strips),
            "total_elastic_body_count": len(self.all_strip_bodies),
            "lower_maximum_centroid_lift_m": self.lower_max_lift_m.tolist(),
            "lower_gripper_contact_frames": self.lower_gripper_contact_frames,
            "release_centroid_z_m": self.release_centroid_z,
            "release_clearance_above_lower_seals_m": self.release_clearance_m,
            "freefall_observed": self.freefall_observed,
            "drop_impact_contact_frames": self.drop_impact_frames,
            "drop_trace": self.drop_trace,
            "motion_trace": self.motion_trace,
            "final_layer_centroid_z_m": [float(body_q[strip["bodies"], 2].mean()) for strip in self.strips],
            "final_contact_matrix": self.contact_counts_host.tolist(),
            "final_weatherstrip_contacts": contact_labels,
            "robot": "FANUC CRX-10iA/L",
            "material_calibrated": False,
            "rest_semi_axes_m": [self.config.weatherstrip.rest_radius_m, self.config.weatherstrip.minor_radius_m],
            "max_centerline_strain_proxy": self.max_centerline_strain,
            "maximum_minimum_segment_lift_m": self.max_lift_height,
            "phases": self.phase_metrics,
            "full_cycle_observed": full_cycle_observed,
            "simulated_time_s": self.sim_time,
            "required_cycle_time_s": self.timeline.total_duration_s,
            "phase": self.last_sample.phase,
            "robot_asset": str(self.config.assets.robot_usd),
            "gripper_asset": str(self.config.assets.gripper_usd),
            "robot_count": 2,
            "gripper_count": 2,
            "robot_body_count": len(self.robot_gripper_bodies) - sum(len(a["gripper_bodies"]) for a in self.assemblies),
            "gripper_body_count": sum(len(a["gripper_bodies"]) for a in self.assemblies),
            "assembly_body_count": len(self.robot_gripper_bodies),
            "assembly_joint_count": len(self.robot_gripper_joints),
            "assembly_joint_coordinate_count": 24,
            "gripper_mimic_constraint_count": 10,
            "maximum_linkage_mimic_error_rad": self.max_mimic_error_rad,
            "added_fingertip_pad_shape_count": 0,
            "weatherstrip_model": "closed rigid-capsule rod solved by Newton VBD",
            "weatherstrip_body_count": len(self.weatherstrip_bodies),
            "weatherstrip_joint_count": len(self.weatherstrip_joints),
            "weatherstrip_shape_count": len(self.weatherstrip_shapes),
            "weatherstrip_density_kg_m3": self.config.weatherstrip.rod_density_kg_m3,
            "weatherstrip_friction": self.config.weatherstrip.friction,
            "table_friction": self.config.scene.table_friction,
            "admm_gripper_weatherstrip_candidate_pairs": self.admm_gripper_weatherstrip_pair_count,
            "left_grip_contact_frames": self.left_grip_contact_frames,
            "right_grip_contact_frames": self.right_grip_contact_frames,
            "commanded_grip_observed": self.saw_commanded_grip,
            "commanded_release_observed": self.saw_commanded_release,
            "jaw_close_observed": self.saw_jaw_close,
            "jaw_reopen_observed": self.saw_jaw_reopen,
            "target_rest_span_m": 2.0 * self.config.weatherstrip.rest_radius_m,
            "target_stretched_span_m": 2.0
            * (self.config.weatherstrip.rest_radius_m + self.config.trajectory.stretch_extension_m),
            "observed_max_weatherstrip_span_m": self.max_weatherstrip_span_m,
            "observed_final_weatherstrip_span_m": float(weatherstrip_extent[:2].max()),
            "observed_final_weatherstrip_vertical_span_m": float(weatherstrip_extent[2]),
            "final_table_support_ceiling_m": table_support_ceiling,
            "final_table_supported_segment_fraction": table_supported_fraction,
            "final_weatherstrip_center_bounds_m": {
                "minimum": weatherstrip_q.min(axis=0).tolist(),
                "maximum": weatherstrip_q.max(axis=0).tolist(),
            },
            "observed_max_tool_target_error_m": self.max_tool_target_error_m,
            "observed_max_locked_grip_distance_m": self.max_grip_distance_m,
            "observed_minimum_weatherstrip_surface_z_m": self.minimum_weatherstrip_surface_z_m,
            "observed_max_jaw_angle_rad": self.max_observed_jaw_angle_rad,
            "final_jaw_angles_rad": joint_q[self.gripper_coords].tolist(),
            "final_weatherstrip_centroid_m": weatherstrip_q.mean(axis=0).tolist(),
            "finite_body_state": bool(np.isfinite(body_q).all()),
            "finite_joint_state": bool(np.isfinite(joint_q).all()),
            "device": str(self.device),
            "solver": "Newton SolverCoupledADMM(MuJoCo, VBD)",
            "grasp_model": "Robotiq 2F-85 mesh collision/friction contact; no kinematic attachment",
        }

    def test_final(self) -> None:
        if isinstance(self.viewer, newton.viewer.ViewerUSD):
            self.viewer.stage.GetRootLayer().Save()
        metrics = self._metrics()
        metrics["validation_passed"] = False
        metrics_path = Path(self.args.metrics_path).expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    key: metrics[key]
                    for key in (
                        "simulated_time_s",
                        "maximum_minimum_segment_lift_m",
                        "freefall_observed",
                        "final_layer_centroid_z_m",
                        "final_second_mean_rms_speed_m_s",
                        "peak_rigid_contacts",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
        if metrics["peak_rigid_contacts"] >= metrics["rigid_contact_capacity"]:
            raise AssertionError("Rigid contact buffer reached capacity")
        if not metrics["finite_body_state"] or not metrics["finite_joint_state"]:
            raise AssertionError("Simulation body or joint state contains NaN/inf")
        if self.admm_gripper_weatherstrip_pair_count <= 0:
            raise AssertionError("No gripper-weatherstrip ADMM candidate pairs")
        if self.minimum_weatherstrip_surface_z_m < -0.05:
            raise AssertionError(f"Weatherstrip fell below the floor: z={self.minimum_weatherstrip_surface_z_m:.4f} m")
        if self.max_tool_target_error_m > 0.12:
            raise AssertionError(f"FANUC gripper tracking error exceeded 0.12 m: {self.max_tool_target_error_m:.4f} m")

        if metrics["full_cycle_observed"]:
            if self.max_lift_height < 0.12:
                raise AssertionError(f"Entire loop was not lifted: {self.max_lift_height:.4f} m")
            if not self.saw_commanded_grip or not self.saw_commanded_release:
                raise AssertionError("Full cycle did not command both grip and release")
            if not self.saw_jaw_close or not self.saw_jaw_reopen:
                raise AssertionError("Physical Robotiq jaws did not close and reopen")
            if self.left_grip_contact_frames <= 0 or self.right_grip_contact_frames <= 0:
                raise AssertionError(
                    "Both Robotiq grippers must make physical weatherstrip contact: "
                    f"left={self.left_grip_contact_frames}, right={self.right_grip_contact_frames}"
                )
            if self.max_grip_distance_m > 0.08:
                raise AssertionError(
                    f"Weatherstrip slipped out of the closed finger pads: distance={self.max_grip_distance_m:.4f} m"
                )
            maximum_span = 1.15 * float(metrics["target_stretched_span_m"])
            if self.max_weatherstrip_span_m > maximum_span:
                raise AssertionError(
                    f"Weatherstrip stretch span {self.max_weatherstrip_span_m:.4f} m "
                    f"exceeded stability limit {maximum_span:.4f} m"
                )
            required_span = 0.90 * float(metrics["target_stretched_span_m"])
            if self.max_weatherstrip_span_m < required_span:
                raise AssertionError(
                    f"Weatherstrip stretch span {self.max_weatherstrip_span_m:.4f} m is below {required_span:.4f} m"
                )
            final_span = float(metrics["observed_final_weatherstrip_span_m"])
            minimum_recovered_span = 0.75 * float(metrics["target_rest_span_m"])
            if final_span < minimum_recovered_span:
                raise AssertionError(
                    f"Weatherstrip remained over-compressed after release: "
                    f"span={final_span:.4f} m, minimum={minimum_recovered_span:.4f} m"
                )
            if final_span > 1.30 * float(metrics["target_rest_span_m"]):
                raise AssertionError(f"Weatherstrip did not elastically recover: final span={final_span:.4f} m")
            final_centroid = metrics["final_weatherstrip_centroid_m"]
            if abs(float(final_centroid[0])) > 0.15 or abs(float(final_centroid[1])) > 0.15:
                raise AssertionError(
                    "Final weatherstrip centroid is outside the central placement region: "
                    f"xy=({float(final_centroid[0]):.4f}, {float(final_centroid[1]):.4f}) m"
                )
            if max(self.lower_max_lift_m) > 0.025:
                raise AssertionError(f"A lower seal was lifted: {self.lower_max_lift_m}")
            if not self.freefall_observed or self.release_clearance_m < 0.05:
                raise AssertionError("Top seal was not released freely above the lower stack")
            if self.drop_impact_frames < 5:
                raise AssertionError("Dropped seal did not land in contact with the lower seals")
            if self.release_centroid_z - float(weatherstrip_q_z := metrics["final_weatherstrip_centroid_m"][2]) < 0.10:
                raise AssertionError("No substantial gravity-driven descent after release")
            if max(metrics["final_second_mean_rms_speed_m_s"]) > 0.003:
                raise AssertionError("Resting stack RMS motion exceeds 3 mm/s in the final second")
            if max(metrics["final_second_max_node_speed_m_s"]) > 0.020:
                raise AssertionError("A resting node exceeds 20 mm/s in the final second")
            if self.contact_counts_host[-1, :2].sum() != 0 or self.contact_counts_host[-1, 3:-1].sum() <= 0:
                raise AssertionError("Final top seal must be released and supported by the lower seals")
            layer_z = metrics["final_layer_centroid_z_m"]
            if layer_z[-1] <= max(layer_z[:-1]) + 0.005:
                raise AssertionError(f"Dropped seal did not settle above both lower seals: {layer_z}")
            if float(metrics["observed_final_weatherstrip_vertical_span_m"]) > 0.09:
                raise AssertionError("Dropped seal did not settle on the stack")
            if not self.config.scene.table_top_z_m < layer_z[-1] < self.config.top_rest_height_m + 0.04:
                raise AssertionError("Dropped seal final height is outside the stack")

        metrics["validation_passed"] = bool(metrics["full_cycle_observed"])
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _preparse_config(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args, _ = parser.parse_known_args(argv)
    return Path(args.config)


def create_parser(config: ProjectConfig) -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument("--config", default=str(config.source_path), help="TOML project configuration.")
    parser.add_argument("--substeps", type=int, default=None, help="Coupled substeps per frame.")
    parser.add_argument("--solver-iterations", type=int, default=None, help="VBD iterations per substep.")
    parser.add_argument("--ik-iterations", type=int, default=None, help="GPU IK iterations per frame.")
    parser.add_argument("--admm-iterations", type=int, default=None, help="ADMM iterations per substep.")
    parser.add_argument("--mujoco-iterations", type=int, default=None, help="MuJoCo iterations per substep.")
    parser.add_argument(
        "--metrics-path",
        default=str(REPO_ROOT / "results" / "metrics.json"),
        help="JSON validation metrics written when --test is enabled.",
    )
    parser.set_defaults(
        device=config.simulation.device,
        num_frames=config.full_cycle_frames,
        output_path=str(REPO_ROOT / "results" / "weatherstrip_cycle.usd"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    config_path = _preparse_config(sys.argv[1:])
    config = load_config(config_path)
    parser = create_parser(config)
    viewer, args = newton.examples.init(parser)
    if isinstance(viewer, newton.viewer.ViewerUSD):
        from .fast_usd import FastUSD

        viewer = FastUSD(args.output_path, fps=config.simulation.fps, num_frames=args.num_frames)
    runtime_config = load_config(args.config)
    simulation = WeatherStripSimulation(viewer, args, runtime_config)
    newton.examples.run(simulation, args)


if __name__ == "__main__":
    main()
