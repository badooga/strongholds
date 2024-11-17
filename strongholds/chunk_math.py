import numpy as np

from . import types, math as gm

stronghold_count = np.array([3, 6, 10, 15, 21, 28, 36, 9])
inner_radii = np.array([1280, 4352, 7424, 10496, 13568, 16640, 19712, 22784])
outer_radii = np.array([2816, 5888, 8960, 12032, 15104, 18176, 21248, 24320])

ring_centers = (inner_radii + outer_radii)/2
ring_radii = np.dstack((inner_radii, outer_radii)).flatten()

def to_radians(y_rot: types.ScalarLike) -> types.ScalarLike:
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
    # rotate by a quarter turn to get to 0 rad in the +x direction
    return gm.angle(1j*z)

def to_yrot(phi: types.ScalarLike) -> types.ScalarLike:
    """
    Converts a polar angle in the xz plane to a
    Minecraft y-rotation value (see `to_radians`).
    """

    # express phi as a unit phasor
    w = np.exp(1j*phi)
    # since w = j*z, z = -j*w
    return gm.angle(1j*w, deg=True)

def snap_chunk(p: types.Coordinates) -> types.Coordinates:
    """
    Snaps coordinates to the northwest corner of the nearest chunk.
    Note that the northwest corner of the *same* chunk can be
    found by using np.floor (rounding down) instead.
    """

    return 16*np.round(p/16)

def in_ring(p: types.Coordinates, ring_num: int | types.Iterable[int]) -> bool:
    """Checks whether coordinates are in the n-th stronghold ring."""

    a, b = inner_radii[ring_num], outer_radii[ring_num]
    return gm.in_interval(gm.radius(p), a, b)

def closest_ring(p: types.Coordinates) -> int | types.Iterable[int]:
    """Finds the stronghold ring that is closest to the given coordinates."""

    radii = np.array(gm.radius(p))
    distances = np.abs(radii[..., None] - ring_radii)
    return distances.argmin(axis=-1) // 2
