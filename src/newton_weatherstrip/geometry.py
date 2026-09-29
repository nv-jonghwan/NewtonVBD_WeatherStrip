"""Pure geometry helpers for the closed VBD weatherstrip."""

from __future__ import annotations

import math


Vec3 = tuple[float, float, float]


def ring_points(segments: int, radius_m: float, height_m: float, minor_radius_m: float | None = None) -> list[Vec3]:
    """Return one point per ring segment, starting at world +X."""
    if segments < 3:
        raise ValueError("segments must be >= 3")
    return [
        (
            radius_m * math.cos(2.0 * math.pi * i / segments),
            (minor_radius_m if minor_radius_m is not None else radius_m) * math.sin(2.0 * math.pi * i / segments),
            height_m,
        )
        for i in range(segments)
    ]


def closed_ring_points(
    segments: int, radius_m: float, height_m: float, minor_radius_m: float | None = None
) -> list[Vec3]:
    """Return N+1 centerline endpoints for Newton add_rod(closed=True)."""
    points = ring_points(segments, radius_m, height_m, minor_radius_m)
    return [*points, points[0]]


def capsule_volume(radius_m: float, cylinder_length_m: float) -> float:
    if radius_m <= 0.0 or cylinder_length_m <= 0.0:
        raise ValueError("capsule dimensions must be positive")
    return math.pi * radius_m**2 * cylinder_length_m + 4.0 * math.pi * radius_m**3 / 3.0


def rod_density(total_mass_kg: float, segments: int, radius_m: float, centerline_length_m: float) -> float:
    if total_mass_kg <= 0.0 or segments < 2:
        raise ValueError("rod mass and segment count must be positive")
    return total_mass_kg / (segments * capsule_volume(radius_m, centerline_length_m))
