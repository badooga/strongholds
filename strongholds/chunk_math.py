import numpy as np

from .math import angle, in_interval, radius
from .types import Coordinates, Iterable, ScalarLike

stronghold_count = np.array([3, 6, 10, 15, 21, 28, 36, 9])
inner_radii = np.array([1280, 4352, 7424, 10496, 13568, 16640, 19712, 22784])
outer_radii = np.array([2816, 5888, 8960, 12032, 15104, 18176, 21248, 24320])
ring_radii = np.dstack((inner_radii, outer_radii)).flatten()

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

    ## we could just do phi = (-90 - y_rot) * pi/180, but
    ## we want to make sure that our result is in [-pi, pi).
    ## we do this by using np.angle, which automatically does that

    # express y_rot as a unit phasor
    z = np.exp(1j*y_rot * np.pi/180)
    # rotate by a quarter turn to get to 0 rad in the +x direction,
    # and then flip vertically to get everything in terms of the xz plane.
    w = np.conj(1j*z)
    return angle(w)

def to_yrot(phi: ScalarLike) -> ScalarLike:
    """
    Converts a polar angle in the xz plane to a
    Minecraft y-rotation value (see `to_radians`).
    """

    # express phi as a unit phasor
    w = np.exp(1j*phi)
    # since w = conj(j*z) = -j * conj(z),
    # we get conj(z) = j*w, so z = conj(j*w)
    z = np.conj(1j*w)
    return angle(z, deg=True)

def snap_chunk(p: Coordinates) -> Coordinates:
    """
    Snaps coordinates to the northwest corner of the nearest chunk.
    Note that the northwest corner of the *same* chunk can be
    found by using np.floor (rounding down) instead.
    """

    return 16*np.round(p/16)

def in_ring(p: Coordinates, ring_num: int) -> bool:
    """Checks whether coordinates are in the n-th stronghold ring."""

    a, b = inner_radii[ring_num], outer_radii[ring_num]
    return in_interval(radius(p), a, b)

def closest_ring(p: Coordinates) -> int | Iterable[int]:
    """Finds the stronghold ring that is closest to the given coordinates."""

    in_rings = [n for n in range(8) if in_ring(p, n)]
    if in_rings:
        return in_rings[0]

    radii = radius(p)
    return np.floor(np.array([np.abs(radii - r) for r in ring_radii]).argmin()/2).astype(int)
