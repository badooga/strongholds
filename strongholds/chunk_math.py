from __future__ import annotations

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
    return gm.np.angle(1j*z)

def to_yrot(phi: types.ScalarLike) -> types.ScalarLike:
    """
    Converts a polar angle in the xz plane to a
    Minecraft y-rotation value (see `to_radians`).
    """

    # express phi as a unit phasor
    w = np.exp(1j*phi)
    # since w = j*z, z = -j*w
    return gm.np.angle(-1j*w, deg=True)

class Coordinates:
    """Stores Minecraft coordinates and its relevant properties."""

    def __init__(self, coords: types.Point | types.Points) -> None:
        self.coords = coords

    def __repr__(self) -> str:
        return str(self.coords)

    @classmethod
    def from_rect(cls, x: types.ScalarLike, z: types.ScalarLike) -> types.Self:
        return cls(x + 1j * z)

    @classmethod
    def from_polar(cls, r: types.ScalarLike, phi: types.ScalarLike) -> types.Self:
        return cls(r * np.exp(1j * phi))

    @classmethod
    def phasor(cls, phi: types.ScalarLike):
        return cls(np.exp(1j * phi))

    @property
    def size(self):
        return self.coords.size

    @property
    def x(self) -> types.ScalarLike:
        return self.coords.real

    @property
    def z(self) -> types.ScalarLike:
        return self.coords.imag

    @property
    def r(self) -> types.ScalarLike:
        return np.abs(self.coords)

    @property
    def phi(self) -> types.ScalarLike:
        return np.angle(self.coords)

    @property
    def yrot(self) -> types.ScalarLike:
        """
        Converts a polar angle in the xz plane to a
        Minecraft y-rotation value (see `to_radians`).
        """

        # express phi as a unit phasor
        w = np.exp(1j*self.phi)
        # since w = j*z, z = -j*w
        return gm.np.angle(1j*w, deg=True)

    @property
    def chunk_corner(self) -> Coordinates:
        return Coordinates(16 * (self.coords // 16))

    @property
    def chunk_center(self) -> Coordinates:
        return self.chunk_corner + (8. + 8.j)

    @property
    def chunk_coords(self) -> Coordinates:
        return Coordinates.from_rect(self.x % 16, self.z % 16)

    def to_xz(self) -> types.CoordinateTuples:
        return np.array([[self.x], [self.z]]).T.squeeze()

    def rotate(self, delta: types.ScalarLike,
               origin: Coordinates | types.Point | types.Points = 0) -> None:
        """
        Rotates a point by delta radians counterclockwise
        about some origin point (defaulting to the origin).
        """

        self.coords = origin + np.exp(1j * delta) * (self.coords - origin)

    def distance(self, other) -> types.ScalarLike:
        try:
            return np.abs(self.coords - other.coords)
        except AttributeError:
            return np.abs(self.coords - other)

    def in_nether(self) -> types.Self:
        return Coordinates(self.coords // 8)

    def in_ring(self, ring_num: int | types.Iterable[int]) -> bool:
        """Checks whether coordinates are in the n-th stronghold ring."""

        a, b = inner_radii[ring_num], outer_radii[ring_num]
        return gm.in_interval(self.r, a, b)

    def closest_ring(self) -> int | types.Iterable[int]:
        """Finds the stronghold ring that is closest to the given coordinates."""

        radii = np.array(self.r)
        distances = np.abs(radii[..., None] - ring_radii)
        return distances.argmin(axis=-1) // 2
