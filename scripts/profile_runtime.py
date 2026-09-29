"""Measure the simulation/CPU-to-USD pipeline, excluding Kit and startup."""

import argparse, json, time, os
import numpy as np
import warp as wp
import newton
from pxr import Usd
from newton_weatherstrip.config import load_config, REPO_ROOT
from newton_weatherstrip.simulation import WeatherStripSimulation

p = argparse.ArgumentParser()
p.add_argument("--frames", type=int, default=90)
p.add_argument("--output", default="results/performance_baseline.json")
cli = p.parse_args()
os.environ["EFORREST_CUDA_GRAPH"] = "1"
from newton_weatherstrip.fast_usd import FastUSD

cfg = load_config()
v = FastUSD(str(REPO_ROOT / "results/profile_scene.usda"), fps=60, num_frames=None, live=True)
a = argparse.Namespace(
    device="cuda:0",
    substeps=None,
    solver_iterations=None,
    ik_iterations=None,
    admm_iterations=None,
    mujoco_iterations=None,
    metrics_path="results/profile_metrics.json",
)
s = WeatherStripSimulation(v, a, cfg)
timings = {k: [] for k in ["targets_ik_ms", "metrics_ms", "step_ms", "surface_ms", "render_ms"]}
measuring = False
for name, key, obj in [
    ("_update_targets", "targets_ik_ms", s),
    ("_collect_metrics", "metrics_ms", s),
    ("update", "surface_ms", s.presentation),
]:
    original = getattr(obj, name)

    def wrapper(*args, _f=original, _key=key, **kw):
        wp.synchronize()
        start = time.perf_counter()
        out = _f(*args, **kw)
        wp.synchronize()
        if measuring:
            timings[_key].append((time.perf_counter() - start) * 1000)
        return out

    setattr(obj, name, wrapper)
for i in range(cli.frames + 10):
    measuring = i >= 10
    t = time.perf_counter()
    s.step()
    wp.synchronize()
    step = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    s.render()
    wp.synchronize()
    render = (time.perf_counter() - t) * 1000
    if measuring:
        timings["step_ms"].append(step)
        timings["render_ms"].append(render)
means = {k: float(np.mean(v)) for k, v in timings.items()}
means["physics_and_other_ms"] = means["step_ms"] - means["targets_ik_ms"] - means["metrics_ms"]
means["usd_other_ms"] = means["render_ms"] - means["surface_ms"]
r = {
    "frames": cli.frames,
    "means": means,
    "pipeline_fps": 1000 / (means["step_ms"] + means["render_ms"]),
    "kit_included": False,
    "substeps": cfg.simulation.substeps,
    "admm_iterations": cfg.simulation.admm_iterations,
}
(REPO_ROOT / cli.output).write_text(json.dumps(r, indent=2) + "\n")
print(json.dumps(r, indent=2))
