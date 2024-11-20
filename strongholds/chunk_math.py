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

    ## we want to make sure that our result is in [-pi, pi).
    ## we do this by using np.angle, which automatically does that

    # express y_rot as a unit phasor
    z = gm.phasor(y_rot, True)
    # rotate by a quarter turn to get to 0 rad in the +x direction
    return gm.np.angle(1j*z)

def to_yrot(phi: types.ScalarLike) -> types.ScalarLike:
    """
    Converts a polar angle in the xz plane to a
    Minecraft y-rotation value (see `to_radians`).
    """

    # express phi as a unit phasor
    w = gm.phasor(phi)
    # since w = j*z, z = -j*w
    return gm.np.angle(-1j*w, deg=True)

class Coordinates:
    """Stores Minecraft coordinates and its relevant properties."""

    def __init__(self, coords: Coordinates | types.PointLike) -> None:
        if isinstance(coords, Coordinates):
            coords = coords.coords
        self.coords: types.PointLike = coords

    def __repr__(self) -> str:
        return str(self.coords)

    def __getitem__(self, mask) -> Coordinates:
        return Coordinates(self.coords[mask])

    @classmethod
    def from_rect(cls, x: types.ScalarLike, z: types.ScalarLike) -> types.Self:
        return cls(x + 1j * z)

    @classmethod
    def from_polar(cls, r: types.ScalarLike,
                   phi: types.ScalarLike,
                   deg: bool = False) -> types.Self:
        return cls(r * gm.phasor(phi, deg=deg))

    @classmethod
    def phasor(cls, phi: types.ScalarLike, deg: bool = False) -> types.Self:
        return cls.from_polar(1, phi, deg)

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
        x, z = self.x // 16, self.z // 16
        return Coordinates.from_rect(16 * x, 16 * z)

    @property
    def chunk_center(self) -> Coordinates:
        return Coordinates(self.chunk_corner.coords + (8. + 8.j))

    @property
    def chunk_coords(self) -> Coordinates:
        return Coordinates.from_rect(self.x % 16, self.z % 16)

    def to_xz(self) -> types.CoordinateTuples:
        return np.array([[self.x], [self.z]]).T.squeeze()

    def rotate(self, delta: types.ScalarLike,
               origin: Coordinates | None = None,
               deg: bool = False) -> None:
        """
        Rotates a point by an angle delta counterclockwise
        about some origin point (defaulting to the origin).
        """
        if origin is None:
            self.coords *= gm.phasor(delta, deg=deg)
        else:
            self.coords = origin.coords + gm.phasor(delta, deg=deg) * (self.coords - origin.coords)

    def distance(self, other) -> types.ScalarLike:
        try:
            return np.abs(self.coords - other.coords)
        except AttributeError:
            return np.abs(self.coords - other)

    def relative_to(self, other) -> Coordinates:
        try:
            rel = self.coords - other.coords
        except AttributeError:
            rel = self.coords - other
        return Coordinates(rel)

    def inner(self, other: Coordinates) -> types.ScalarLike:
        return (self.coords.conj() * other.coords).real

    def outer(self, other: Coordinates) -> types.ScalarLike:
        return (self.coords.conj() * other.coords).imag

    def in_nether(self) -> Coordinates:
        return Coordinates.from_rect(self.x // 8, self.z // 8)

    def in_ring(self, ring_num: int | types.Iterable[int]) -> bool:
        """Checks whether coordinates are in the n-th stronghold ring."""

        a, b = inner_radii[ring_num], outer_radii[ring_num]
        return gm.in_interval(self.r, a, b)

    def closest_ring(self) -> int | types.Iterable[int]:
        """Finds the stronghold ring that is closest to the given coordinates."""

        radii = np.array(self.r)
        distances = np.abs(radii[..., None] - ring_radii)
        return distances.argmin(axis=-1) // 2
