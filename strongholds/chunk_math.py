import numpy as np

from .geometry import distance, in_interval
from .types import RectCoordinates

stronghold_count = (3, 6, 10, 15, 21, 28, 36, 9)
inner_radii = (1280, 4352, 7424, 10496, 13568, 16640, 19712, 22784)
outer_radii = (2816, 5888, 8960, 12032, 15104, 18176, 21248, 24320)
ring_radii = sum(zip(inner_radii, outer_radii), ())

def snap_chunk(p: RectCoordinates) -> RectCoordinates:
    """Snaps coordinates to the northwest corner of the nearest chunk."""

    return 16*np.floor(p/16)

def in_ring(p: RectCoordinates, ring_num: int) -> bool:
    """Checks whether coordinates are in the n-th stronghold ring."""

    a, b = inner_radii[ring_num], outer_radii[ring_num]
    radii = distance(p, 0)
    return in_interval(radii, a, b)

def closest_ring(p: RectCoordinates) -> int | list[int]:
    """Finds the stronghold ring that is closest to the given coordinates."""

    in_rings = [n for n in range(8) if in_ring(p, n)]
    if in_rings:
        return in_rings[0]

    radii = distance(p, 0)
    return np.floor(np.array([np.abs(radii - r) for r in ring_radii]).argmin()/2)
