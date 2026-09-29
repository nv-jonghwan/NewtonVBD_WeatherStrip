"""Guard the public asset download boundary and pinned source integrity."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_assets", ROOT / "scripts/fetch_assets.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class AssetManifestTest(unittest.TestCase):
    def test_public_manifest_has_unique_relative_paths_and_hashes(self):
        manifest = json.loads((ROOT / "assets/manifest.json").read_text())
        paths = [f["path"] for f in manifest["files"]]
        self.assertEqual(len(paths), len(set(paths)))
        for item in manifest["files"]:
            self.assertTrue(item["path"].startswith(("assets/fanuc/", "assets/robotiq_2f85/")))
            self.assertNotIn("..", Path(item["path"]).parts)
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(item["url"].startswith("https://"))

    def test_corrupt_cached_asset_is_not_replaced_or_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "assets").mkdir()
            target = root / "assets/bad.usd"
            target.write_bytes(b"corrupt")
            manifest = {
                "files": [
                    {
                        "path": "assets/bad.usd",
                        "url": "https://example.invalid/asset",
                        "sha256": hashlib.sha256(b"expected").hexdigest(),
                    }
                ]
            }
            (root / "assets/manifest.json").write_text(json.dumps(manifest))
            with patch.object(fetch, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    fetch.fetch_assets(root, offline=True)
            self.assertEqual(target.read_bytes(), b"corrupt")


if __name__ == "__main__":
    unittest.main()
