"""Encode a validated Isaac Lab viewport recording with FFmpeg (no Python extras)."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("capture_dir", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--width", type=int, default=960)
args = parser.parse_args()
if args.width < 1:
    parser.error("--width must be positive")
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    parser.error("FFmpeg is required (Ubuntu: sudo apt install ffmpeg)")
manifest = json.loads((args.capture_dir / "capture.json").read_text())
if not manifest["completed"] or not manifest["validation_passed"]:
    parser.error("Only a completed, validated simulation recording can be published")
frames = manifest["frames"]
if len(frames) < 2:
    parser.error("The recording must have multiple frames")
for i, frame in enumerate(frames):
    expected = f"frame_{i:05d}.png"
    if frame["file"] != expected or not (args.capture_dir / expected).is_file():
        parser.error(f"Missing or nonsequential capture: {expected}")
args.output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-n",
        "-framerate",
        str(manifest["playback_fps"]),
        "-i",
        str(args.capture_dir / "frame_%05d.png"),
        "-filter_complex",
        f"scale={args.width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
        "-frames:v",
        str(len(frames)),
        "-loop",
        "0",
        str(args.output),
    ],
    check=True,
)
print(
    f"{args.output}: {len(frames)} frames, {manifest['playback_fps']:g} FPS, "
    f"{args.output.stat().st_size / 1024**2:.2f} MiB"
)
