import numpy as np

from .math import in_interval, radius, rectpoint, to_xz
from .types import Coordinates, ScalarLike

stronghold_count = (3, 6, 10, 15, 21, 28, 36, 9)
inner_radii = (1280, 4352, 7424, 10496, 13568, 16640, 19712, 22784)
outer_radii = (2816, 5888, 8960, 12032, 15104, 18176, 21248, 24320)
ring_radii = sum(zip(inner_radii, outer_radii), ())

def to_radians(y_rot: ScalarLike) -> ScalarLike:
    """
    Converts Minecraft's y-rotation value to a polar angle in radians.
    
    In Minecraft, the +x and +z directions correspond to east and south (not north!),
    respectively. The angle y_rot is 0 at due south, while what I define to be the
    polar angle phi is 0 at due east. Both increase in the clockwise cardinal direction
    and are in the range (-pi, pi] (or (-180, 180] for degrees).

    For our purposes, we do everything in terms of x and z, and
    use `Ax.invert_yaxis()` to create an accurate image when plotting.
    """

    # express y_rot as a unit phasor
    z = np.exp(1j*y_rot * np.pi/180)
    # rotate by a quarter turn to get to 0 rad in the +x direction,
    # and then flip vertically to get everything in terms of the xz plane.
    w = np.conj(1j*z)
    return np.angle(w)

def snap_chunk(p: Coordinates) -> Coordinates:
    """Snaps coordinates to the northwest corner of the nearest chunk."""

    # np.floor doesn't work on complex numbers, so
    # it's necessary to convert to [x, z] form first
    p_new = 16*np.floor(to_xz(p)/16)
    return rectpoint(*p_new.T)

def in_ring(p: Coordinates, ring_num: int) -> bool:
    """Checks whether coordinates are in the n-th stronghold ring."""

    a, b = inner_radii[ring_num], outer_radii[ring_num]
    return in_interval(radius(p), a, b)

def closest_ring(p: Coordinates) -> int | list[int]:
    """Finds the stronghold ring that is closest to the given coordinates."""

    in_rings = [n for n in range(8) if in_ring(p, n)]
    if in_rings:
        return in_rings[0]

    radii = radius(p)
    return np.floor(np.array([np.abs(radii - r) for r in ring_radii]).argmin()/2)
