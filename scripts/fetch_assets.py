"""Fetch only the pinned robot files listed in assets/manifest.json; verify SHA-256."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def fetch_assets(root: Path, *, offline: bool = False) -> int:
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    root = root.resolve()
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root / "assets"):
            raise ValueError(f"Asset path escapes destination: {item['path']}")
        if path.exists():
            data = path.read_bytes()
        elif offline:
            raise FileNotFoundError(f"Missing {item['path']}; run scripts/fetch_assets.py online first")
        else:
            request = Request(item["url"], headers={"User-Agent": "NewtonVBD-WeatherStrip/0.2"})
            with urlopen(request, timeout=90) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {item['path']}; refusing to replace or use this file")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(data)
            temporary.replace(path)
        print(f"VERIFIED {item['path']}", flush=True)
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Verify existing assets without network access")
    parser.add_argument("--root", type=Path, default=ROOT, help="Checkout receiving the assets")
    args = parser.parse_args()
    print(f"Verified {fetch_assets(args.root, offline=args.offline)} asset files.")


if __name__ == "__main__":
    main()
