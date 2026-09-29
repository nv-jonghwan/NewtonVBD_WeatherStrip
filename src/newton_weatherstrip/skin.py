"""GPU spline skinning: no Python per-ring frame transport or physics changes."""

import math
import warp as wp


@wp.func
def perpendicular(t: wp.vec3):
    axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(t[1]) < wp.abs(t[0]) and wp.abs(t[1]) <= wp.abs(t[2]):
        axis = wp.vec3(0.0, 1.0, 0.0)
    elif wp.abs(t[2]) < wp.abs(t[0]) and wp.abs(t[2]) < wp.abs(t[1]):
        axis = wp.vec3(0.0, 0.0, 1.0)
    return wp.normalize(wp.cross(t, axis))


@wp.func
def transport(n: wp.vec3, prev: wp.vec3, curr: wp.vec3):
    axis = wp.cross(prev, curr)
    c = wp.clamp(wp.dot(prev, curr), -1.0, 1.0)
    result = n
    if c > -0.999999:
        result = n + wp.cross(axis, n) + wp.cross(axis, wp.cross(axis, n)) / (1.0 + c)
    result = result - curr * wp.dot(curr, result)
    if wp.length(result) < 1.0e-10:
        result = perpendicular(curr)
    return wp.normalize(result)


@wp.kernel
def interpolate(
    body_q: wp.array(dtype=wp.transform),
    ids: wp.array2d(dtype=int),
    count: int,
    subdivisions: int,
    centers: wp.array2d(dtype=wp.vec3),
):
    layer, ring = wp.tid()
    i = ring // subdivisions
    t = float(ring % subdivisions) / float(subdivisions)
    p0 = wp.transform_get_translation(body_q[ids[layer, (i + count - 1) % count]])
    p1 = wp.transform_get_translation(body_q[ids[layer, i]])
    p2 = wp.transform_get_translation(body_q[ids[layer, (i + 1) % count]])
    p3 = wp.transform_get_translation(body_q[ids[layer, (i + 2) % count]])
    centers[layer, ring] = 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


@wp.kernel
def frames(
    centers: wp.array2d(dtype=wp.vec3),
    count: int,
    tangent: wp.array2d(dtype=wp.vec3),
    normal: wp.array2d(dtype=wp.vec3),
    twist: wp.array(dtype=float),
):
    layer = wp.tid()
    for i in range(count):
        tangent[layer, i] = wp.normalize(centers[layer, (i + 1) % count] - centers[layer, (i + count - 1) % count])
    normal[layer, 0] = perpendicular(tangent[layer, 0])
    for i in range(1, count):
        normal[layer, i] = transport(normal[layer, i - 1], tangent[layer, i - 1], tangent[layer, i])
    closing = transport(normal[layer, count - 1], tangent[layer, count - 1], tangent[layer, 0])
    twist[layer] = wp.atan2(
        wp.dot(wp.cross(closing, normal[layer, 0]), tangent[layer, 0]), wp.dot(closing, normal[layer, 0])
    )


@wp.kernel
def vertices(
    centers: wp.array2d(dtype=wp.vec3),
    tangent: wp.array2d(dtype=wp.vec3),
    normal: wp.array2d(dtype=wp.vec3),
    twist: wp.array(dtype=float),
    count: int,
    sides: int,
    radius: float,
    output: wp.array3d(dtype=wp.vec3),
):
    layer, i, j = wp.tid()
    correction = float(i) * twist[layer] / float(count)
    n = normal[layer, i] * wp.cos(correction) + wp.cross(tangent[layer, i], normal[layer, i]) * wp.sin(correction)
    b = wp.cross(tangent[layer, i], n)
    angle = 2.0 * wp.pi * float(j) / float(sides)
    radial = n * wp.cos(angle) + b * wp.sin(angle)
    output[layer, i * sides + j, 0] = centers[layer, i] + radius * radial
    output[layer, i * sides + j, 1] = radial


class TubeSkin:
    def __init__(self, body_ids, radius, device, subdivisions=4, sides=16):
        self.layers = len(body_ids)
        self.count = len(body_ids[0])
        self.rings = self.count * subdivisions
        self.subdivisions = subdivisions
        self.sides = sides
        self.radius = radius
        self.device = device
        self.ids = wp.array(body_ids, dtype=int, device=device)
        self.centers = wp.empty((self.layers, self.rings), dtype=wp.vec3, device=device)
        self.tangent = wp.empty_like(self.centers)
        self.normal = wp.empty_like(self.centers)
        self.twist = wp.empty(self.layers, dtype=float, device=device)
        self.output = wp.empty((self.layers, self.rings * sides, 2), dtype=wp.vec3, device=device)

    def update(self, body_q):
        wp.launch(
            interpolate,
            dim=(self.layers, self.rings),
            inputs=[body_q, self.ids, self.count, self.subdivisions, self.centers],
            device=self.device,
        )
        wp.launch(
            frames,
            dim=self.layers,
            inputs=[self.centers, self.rings, self.tangent, self.normal, self.twist],
            device=self.device,
        )
        wp.launch(
            vertices,
            dim=(self.layers, self.rings, self.sides),
            inputs=[
                self.centers,
                self.tangent,
                self.normal,
                self.twist,
                self.rings,
                self.sides,
                self.radius,
                self.output,
            ],
            device=self.device,
        )
        return self.output.numpy()
