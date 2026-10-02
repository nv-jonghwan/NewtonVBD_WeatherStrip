"""Isaac Lab AppLauncher + live Newton coupled solver USD bridge.

The custom coupled solver is stepped explicitly; PhysX never advances this scene.
The standard timeline Play/Pause and the WeatherStrip panel control the same clock.
"""

import argparse
import asyncio
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Newton VBD: dual FANUC rubber weatherstrip")
parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/default.toml"))
parser.add_argument("--auto-play", action="store_true")
parser.add_argument("--pause-at", type=float, default=None)
parser.add_argument("--smoke-frames", type=int, default=0)
parser.add_argument("--capture-dir", type=Path, help="Save a complete cycle as numbered viewport PNGs.")
parser.add_argument("--capture-stride", type=int, default=4, help="Physics frames per capture (60/4 = 15 FPS).")
parser.add_argument("--camera", choices=("scene", "close"), default="scene")
parser.add_argument("--exit-after-cycle", action="store_true")
AppLauncher.add_app_launcher_args(parser)
cli = parser.parse_args()
if cli.capture_stride < 1:
    parser.error("--capture-stride must be positive")
if cli.capture_dir:
    cli.capture_dir = cli.capture_dir.resolve()
    if cli.capture_dir.exists() and any(cli.capture_dir.iterdir()):
        parser.error("--capture-dir must be empty to avoid mixing recordings")
    cli.capture_dir.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(cli)
app = launcher.app

import json
import time
import numpy as np
import omni.timeline
import omni.usd
import omni.ui as ui
from pxr import Gf, Usd, UsdGeom, UsdUtils
import newton
import warp as wp
from newton_weatherstrip.config import load_config, REPO_ROOT
from newton_weatherstrip.simulation import WeatherStripSimulation
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file


from newton_weatherstrip.fast_usd import FastUSD

cfg = load_config(cli.config)
viewer = FastUSD(str(REPO_ROOT / "results/live_scene.usda"), fps=cfg.simulation.fps, num_frames=None, live=True)
args = argparse.Namespace(
    device=cli.device,
    substeps=None,
    solver_iterations=None,
    ik_iterations=None,
    admm_iterations=None,
    mujoco_iterations=None,
    metrics_path=str(REPO_ROOT / "results/gui_metrics.json"),
)
with wp.ScopedDevice(cli.device):
    sim = WeatherStripSimulation(viewer, args, cfg)
    sim.render()
viewer.stage.GetRootLayer().Save()
cache = UsdUtils.StageCache.Get()
cache.Insert(viewer.stage)
omni.usd.get_context().attach_stage_with_callback(cache.GetId(viewer.stage).ToLongInt())
for _ in range(20):
    app.update()


def set_camera(close=False):
    camera = UsdGeom.Camera.Define(viewer.stage, "/root/Camera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000))
    camera.CreateFocalLengthAttr(45.0 if close else 27.0)
    eye = Gf.Vec3d(1.1, -1.25, 1.25) if close else Gf.Vec3d(1.9, -2.25, 1.65)
    view = Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(0, 0, 0.50), Gf.Vec3d(0, 0, 1))
    xform = UsdGeom.Xformable(camera)
    ops = xform.GetOrderedXformOps()
    op = ops[0] if ops else xform.AddTransformOp()
    op.Set(view.GetInverse())
    viewport = get_active_viewport()
    if viewport:
        viewport.camera_path = "/root/Camera"


set_camera(cli.camera == "close")
viewport = get_active_viewport()
timeline = omni.timeline.get_timeline_interface()
timeline.set_end_time(cfg.trajectory.total_duration_s + 1)
timeline.set_looping(False)
timeline.set_auto_update(False)
timeline.pause()
initial_q = sim.state_0.body_q.numpy().copy()
initial_joint_q = sim.state_0.joint_q.numpy().copy()
reset_requested = False
completed = False
validation_passed = None


def reset():
    global reset_requested
    timeline.pause()
    reset_requested = True


def play():
    if completed:
        reset()
    timeline.play()


window = ui.Window("Newton VBD | WeatherStrip", width=460, height=340)
with window.frame:
    with ui.VStack(spacing=8):
        ui.Label("Dual FANUC CRX-10iA/L + Robotiq 2F-85", height=26)
        ui.Label("3 seals | Pick top > Stretch > Sag > Open > Drop", height=22)
        with ui.HStack(height=35, spacing=8):
            ui.Button("Play", clicked_fn=play)
            ui.Button("Pause", clicked_fn=timeline.pause)
            ui.Button("Reset", clicked_fn=reset)
        with ui.HStack(height=28, spacing=8):
            ui.Button("Scene view", clicked_fn=lambda: set_camera(False))
            ui.Button("Seal close-up", clicked_fn=lambda: set_camera(True))

        def capture_screenshot():
            global capture
            name = time.strftime("weatherstrip_%Y%m%d_%H%M%S.png")
            active = get_active_viewport()
            if active:
                capture = capture_viewport_to_file(active, str(REPO_ROOT / "results" / name))

        ui.Button("Save screenshot", clicked_fn=capture_screenshot, height=28)
        status = ui.Label("Ready - press Play", height=30)
        span_label = ui.Label("Top seal span: 0.560 m", height=25)
        fps_label = ui.Label("FPS: measuring", height=25)
        ui.Label("Black rubber | illustrative material, not calibrated", height=25)

capture_frames = []


def capture_frame():
    """Finish rendering this physical state before advancing the solver again."""
    set_camera(cli.camera == "close")
    active = get_active_viewport()
    if active is None:
        raise RuntimeError("Viewport capture requires a displayed Kit viewport")
    filename = f"frame_{len(capture_frames):05d}.png"
    destination = cli.capture_dir / filename
    request = capture_viewport_to_file(active, str(destination))
    pending = asyncio.ensure_future(request.wait_for_result())
    deadline = time.monotonic() + 30.0
    while not pending.done() or not destination.exists():
        if not app.is_running() or time.monotonic() > deadline:
            pending.cancel()
            raise RuntimeError(f"Viewport capture did not finish: {destination}")
        app.update()
    pending.result()
    capture_frames.append(dict(file=filename, sim_time_s=sim.sim_time, phase=sim.last_sample.phase))
    (cli.capture_dir / "capture.json").write_text(
        json.dumps(
            {
                "newton": newton.__version__,
                "warp": wp.__version__,
                "physics_fps": cfg.simulation.fps,
                "stride": cli.capture_stride,
                "playback_fps": cfg.simulation.fps / cli.capture_stride,
                "camera": cli.camera,
                "completed": completed,
                "validation_passed": validation_passed,
                "frames": capture_frames,
            },
            indent=2,
        )
        + "\n"
    )


if cli.capture_dir:
    for _ in range(20):
        app.update()
    capture_frame()
if cli.auto_play or cli.smoke_frames or cli.capture_dir:
    timeline.play()
print(
    "WEATHERSTRIP_GUI_READY: Isaac Lab AppLauncher; live Newton coupled physics; paused="
    + str(not timeline.is_playing()),
    flush=True,
)
frames = 0
ticks = 0
capture = None
perf_frames = []
perf_report = {}
control_path = REPO_ROOT / "results/gui_control.json"
try:
    while app.is_running():
        frame_started = time.perf_counter()
        step_ms = render_ms = 0.0
        advanced = False
        if reset_requested:
            # Reconstruct solver as well as state, so no ADMM/contact history survives reset.
            was_playing = timeline.is_playing()
            timeline.pause()
            cfg = load_config(cli.config)
            viewer.clear_model()
            with wp.ScopedDevice(cli.device):
                sim = WeatherStripSimulation(viewer, args, cfg)
            sim.render()
            set_camera()
            timeline.set_current_time(0)
            reset_requested = False
            perf_frames = []
            perf_report = {}
            fps_label.text = "FPS: measuring"
            timeline.set_end_time(cfg.trajectory.total_duration_s + 1)
            completed = False
            validation_passed = None
            status.text = "Ready - press Play"
            span_label.text = f"Span: {2 * cfg.weatherstrip.rest_radius_m:.3f} m"
            if was_playing:
                timeline.play()
        if timeline.is_playing() and not completed:
            with wp.ScopedDevice(cli.device):
                started = time.perf_counter()
                sim.step()
                step_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                sim.render()
                render_ms = (time.perf_counter() - started) * 1000
                advanced = True
            frames += 1
            timeline.set_current_time(sim.sim_time)
            if cli.pause_at is not None and sim.sim_time >= cli.pause_at:
                timeline.pause()
                cli.pause_at = None
            if sim.sim_time >= sim.timeline.total_duration_s:
                completed = True
                timeline.pause()
                try:
                    sim.test_final()
                    validation_passed = True
                    status.text = "Complete - top seal picked and dropped on stack"
                except AssertionError as exc:
                    validation_passed = False
                    status.text = "Validation needs review: " + str(exc)
                    print("WEATHERSTRIP_VALIDATION_FAILED: " + str(exc), flush=True)
            else:
                status.text = (
                    f"{sim.last_sample.phase.upper()}   {sim.sim_time:.2f} / {sim.timeline.total_duration_s:.1f} s"
                )
            centers = sim.state_0.body_q.numpy()[sim.weatherstrip_bodies, :3]
            span_label.text = f"Span: {np.ptp(centers[:, 0]):.3f} m   Max: {sim.max_weatherstrip_span_m:.3f} m"
        kit_started = time.perf_counter()
        app.update()
        if advanced:
            elapsed_ms = (time.perf_counter() - frame_started) * 1000
            kit_ms = (time.perf_counter() - kit_started) * 1000
            perf_frames.append([step_ms, render_ms, kit_ms, elapsed_ms])
            if len(perf_frames) >= 30:
                recent = np.asarray(perf_frames[-60:])
                means = recent.mean(axis=0)
                perf_report = dict(
                    fps=1000 / means[3],
                    physics_and_metrics_ms=means[0],
                    usd_update_ms=means[1],
                    kit_ms=means[2],
                    simulated_seconds_per_wall_second=(1000 / means[3]) / cfg.simulation.fps,
                )
                fps_label.text = f"FPS: {perf_report['fps']:.1f} | Physics: {means[0]:.1f} ms | USD: {means[1]:.1f} ms"
        if cli.capture_dir and advanced and (frames % cli.capture_stride == 0 or completed):
            capture_frame()
        ticks += 1
        if control_path.exists():
            command = json.loads(control_path.read_text())
            control_path.unlink()
            action = command.get("action")
            if action == "play":
                play()
            elif action == "pause":
                timeline.pause()
            elif action == "reset":
                reset()
            elif action == "camera":
                set_camera(command.get("view") == "close")
            elif action == "capture" and viewport:
                filename = Path(command.get("filename", "isaaclab_preview.png")).name
                capture = capture_viewport_to_file(viewport, str(REPO_ROOT / "results" / filename))
        if ticks % 30 == 0:
            (REPO_ROOT / "results/gui_status.json").write_text(
                json.dumps(
                    {
                        "newton": newton.__version__,
                        "warp": wp.__version__,
                        "sim_time_s": sim.sim_time,
                        "phase": sim.last_sample.phase,
                        "playing": timeline.is_playing(),
                        "completed": completed,
                        "validation_passed": validation_passed,
                        "gripper": "Robotiq 2F-85",
                        "stack_count": len(sim.strips),
                        "performance": perf_report,
                        "frames": frames,
                        "ticks": ticks,
                        "cuda_graph": sim.physics_graph is not None,
                        "finite": bool(np.isfinite(sim.state_0.body_q.numpy()).all()),
                    },
                    indent=2,
                )
                + "\n"
            )
        if completed and perf_frames:
            (REPO_ROOT / "results/gui_performance.json").write_text(
                json.dumps(
                    {
                        "capture_enabled": bool(cli.capture_dir),
                        "last_window": perf_report,
                        "sampled_frames": len(perf_frames),
                        "mean_ms": np.asarray(perf_frames).mean(axis=0).tolist(),
                    },
                    indent=2,
                )
                + "\n"
            )
            perf_frames = []
        if completed and cli.exit_after_cycle:
            if not validation_passed:
                raise RuntimeError("GUI cycle validation failed")
            break
        if cli.smoke_frames and frames >= cli.smoke_frames:
            report = {
                "isaaclab_app_launcher": True,
                "newton_live_steps": frames,
                "stage_prims": sum(1 for _ in viewer.stage.Traverse()),
                "finite": bool(np.isfinite(sim.state_0.body_q.numpy()).all()),
                "physics_backend": "custom Newton SolverCoupledADMM bridge",
            }
            (REPO_ROOT / "results/isaaclab_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
            break
        if not timeline.is_playing():
            time.sleep(0.01)
    if cli.capture_dir and not completed:
        raise RuntimeError("GUI closed before the recording completed")
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    app.close()
