"""Regression checks for the closed, smooth skin at sharply bent grasp regions."""

import unittest
from collections import Counter
import numpy as np
from pxr import Usd
from newton_weatherstrip.presentation import create_tube, smooth_centerline, tube_geometry


class SurfaceTest(unittest.TestCase):
    def test_bent_loop_is_closed_and_smooth(self):
        t = np.arange(64) * 2 * np.pi / 64
        centers = np.column_stack((0.36 * np.cos(t), 0.15 * np.sin(t), 0.12 * np.cos(2 * t)))
        original = centers.copy()
        skin = smooth_centerline(centers)
        np.testing.assert_array_equal(centers, original)
        np.testing.assert_allclose(skin[::4], centers, atol=1e-12)
        points, normals = tube_geometry(centers, 0.010)
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2e-6)
        np.testing.assert_allclose(
            np.linalg.norm(points.reshape(256, 16, 3) - skin[:, None, :], axis=2), 0.010, atol=1e-7
        )
        # Normals remain aligned across adjacent rings, including the periodic seam.
        n = normals.reshape(256, 16, 3)
        self.assertGreater(float(np.min(np.sum(n * np.roll(n, 1, axis=0), axis=2))), 0.96)
        stage = Usd.Stage.CreateInMemory()
        mesh = create_tube(stage, centers, 0.010)
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 4)
        edges = Counter(tuple(sorted((int(a), int(b)))) for f in faces for a, b in zip(f, np.roll(f, -1)))
        self.assertTrue(all(v == 2 for v in edges.values()))
        self.assertEqual(mesh.GetNormalsInterpolation(), "vertex")


class FastSurfaceTest(unittest.TestCase):
    def test_warp_skin_matches_reference(self):
        import warp as wp
        from newton_weatherstrip.skin import TubeSkin

        t = np.arange(64) * 2 * np.pi / 64
        centers = np.column_stack((0.36 * np.cos(t), 0.15 * np.sin(t), 0.12 * np.cos(2 * t)))
        body = np.zeros((64, 7), np.float32)
        body[:, :3] = centers
        body[:, 6] = 1
        skin = TubeSkin([list(range(64))], 0.01, "cpu")
        actual = skin.update(wp.array(body, dtype=wp.transform, device="cpu"))[0]
        points, normals = tube_geometry(centers, 0.01)
        np.testing.assert_allclose(actual[:, 0], points, atol=3e-6)
        np.testing.assert_allclose(actual[:, 1], normals, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
