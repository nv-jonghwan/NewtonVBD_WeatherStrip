"""Single-world USD bridge; static appearance authored once, dynamic poses in bulk.

ViewerUSD's generic path copies every shape batch separately and rewrites static
colors/scales/visibility. This scene only changes rigid poses and the seal skins.
"""

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom
import newton


class FastUSD(newton.viewer.ViewerUSD):
    def __init__(self, *args, live=False, **kwargs):
        self.live = live
        self._pose_cache = None
        self.body_q_host = None
        super().__init__(*args, **kwargs)

    def begin_frame(self, time):
        if self.live:
            self._frame_index = Usd.TimeCode.Default()
            self._frame_count += 1
        else:
            super().begin_frame(time)

    def clear_model(self):
        self._pose_cache = None
        self.body_q_host = None
        super().clear_model()

    def log_state(self, state):
        if self._pose_cache is None:
            super().log_state(state)
            entries = []
            shape_ids = []
            for batch in self._shape_instances.values():
                if not self._should_show_shape(batch.flags, batch.static, batch.geo_type):
                    continue
                for i, shape in enumerate(batch.model_shapes):
                    if int(self.model.shape_body.numpy()[int(shape)]) < 0:
                        continue
                    prim = self.stage.GetPrimAtPath(self._get_path(batch.name) + f"/instance_{i}")
                    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
                    entries.append((ops[0], ops[1]))
                    shape_ids.append(int(shape))
            self._pose_cache = entries
            self._shape_local = self.model.shape_transform.numpy()[shape_ids]
            self._shape_parents = self.model.shape_body.numpy()[shape_ids]
            return
        body = self.body_q_host if self.body_q_host is not None else state.body_q.numpy()
        parent = body[self._shape_parents]
        local = self._shape_local
        # Compose xyzw quaternions and rotate translations in one NumPy batch.
        a, b = parent[:, 3:6], local[:, 3:6]
        aw, bw = parent[:, 6:7], local[:, 6:7]
        qxyz = aw * b + bw * a + np.cross(a, b)
        qw = aw * bw - np.sum(a * b, axis=1, keepdims=True)
        v = local[:, :3]
        twice = 2 * np.cross(a, v)
        pos = parent[:, :3] + v + aw * twice + np.cross(a, twice)
        with Sdf.ChangeBlock():
            for i, (translate, orient) in enumerate(self._pose_cache):
                translate.Set(Gf.Vec3d(*map(float, pos[i])), self._frame_index)
                orient.Set(Gf.Quatf(float(qw[i, 0]), *map(float, qxyz[i])), self._frame_index)
