from __future__ import annotations

import numpy as np

from . import types, math as gm

stronghold_count = np.array([3, 6, 10, 15, 21, 28, 36, 9])
inner_radii = np.array([1280, 4352, 7424, 10496, 13568, 16640, 19712, 22784])
outer_radii = np.array([2816, 5888, 8960, 12032, 15104, 18176, 21248, 24320])

ring_centers = (inner_radii + outer_radii) / 2
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

    # we want to make sure that our result is in [-pi, pi).
    # we do this by using np.angle, which automatically does that

    # express y_rot as a unit phasor
    z = gm.phasor(y_rot, True)
    # rotate by a quarter turn to get to 0 rad in the +x direction
    return gm.np.angle(1j * z)


def to_yrot(phi: types.ScalarLike) -> types.ScalarLike:
    """
    Converts a polar angle in the xz plane to a
    Minecraft y-rotation value (see `to_radians`).
    """

    # express phi as a unit phasor
    w = gm.phasor(phi)
    # since w = j*z, z = -j*w
    return gm.np.angle(-1j * w, deg=True)


class Coordinates(np.ndarray):
    """Stores Minecraft coordinates and its relevant properties."""

    def __new__(cls, coords: Coordinates | types.PointLike) -> types.Self:
        """Constructs the Coordinates from complex data.

        Args:
            coords (Coordinates | PointLike): The coordinates in complex form.
        """

        obj = np.asarray(coords, dtype=np.complex128).view(cls)
        return obj

    @classmethod
    def from_rect(cls, x: types.ScalarLike, z: types.ScalarLike) -> types.Self:
        """Constructs the Coordinates from xz positions.

        Args:
            x (ScalarLike): The x coordinates.
            z (ScalarLike): The z coordinates.
        """

        return cls(x + 1j * z)

    @classmethod
    def from_polar(cls, r: types.ScalarLike,
                   phi: types.ScalarLike,
                   deg: bool = False) -> types.Self:
        """Constructs the Coordinates from the radius and x/z angle.

        Args:
            r (ScalarLike): The distance from the origin.
            phi (ScalarLike): The polar angle between the x and z axes.
            deg (bool, optional): Whether to consider phi in degrees. Defaults to False.
        """

        return cls(r * gm.phasor(phi, deg=deg))

    @classmethod
    def from_chunk(cls, cx: types.ScalarLike,
                   cz: types.ScalarLike,
                   center: bool = False) -> types.Self:
        """Constructs the Coordinates from the chunk coordinates.

        Args:
            cx (ScalarLike): The chunk number along the x axis.
            cz (ScalarLike): The chunk number along the z axis.
            center (bool, optional): Whether to return coordinates at the (8, 8)
            coordinates of the chunk (True) or at (0, 0) in the chunk (False).
            Defaults to False.
        """

        n = 8 if center else 0
        return cls.from_rect(16*cx + n, 16*cz + 8)

    @property
    def coords(self) -> types.PointLike:
        """Gets the underlying data of the array.

        Returns:
            PointLike: A regular NumPy array of the coordinates.
        """

        coords = self.__array__()
        if not coords.shape:
            coords = coords.item()
        return np.complex128(coords)

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
        """Converts a polar angle in the xz plane to a
        Minecraft y-rotation value (see to_radians).
        """

        # express phi as a unit phasor
        w = gm.phasor(self.phi)
        # since w = j*z, z = -j*w
        return gm.np.angle(1j * w, deg=True)

    @property
    def chunk_corner(self) -> types.Self:
        x, z = self.x // 16, self.z // 16
        return self.__class__.from_rect(16 * x, 16 * z)

    @property
    def chunk_center(self) -> types.Self:
        return self.__class__(self.chunk_corner + (8.0 + 8.0j))

    @property
    def chunk_coords(self) -> types.Self:
        return self.__class__.from_rect(self.x // 16, self.z // 16)

    def to_xz(self):
        return np.stack((self.x, self.y), -1)

    def rotate(self, delta: types.ScalarLike,
               origin: Coordinates | None = None,
               deg: bool = False) -> None:
        """Rotates the Coordinates by an angle delta about some origin point.

        Args:
            delta (ScalarLike): The counterclockwise angle to rotate by in the xz plane.
            origin (Coordinates | None, optional): The point to rotate
            around. Defaults to None (the origin).
            deg (bool, optional): Whether to consider delta in degrees. Defaults to False.
        """

        if origin is None:
            origin = Coordinates(0)

        self = origin + gm.phasor(delta, deg=deg) * (self - origin)

    def relative_angle(self, other, direction: types.Self | None = None) -> types.ScalarLike:
        rel_phi = (self - other).phi
        if direction is not None:
            rel_phi -= direction.phi
        return rel_phi

    def inner(self, other) -> types.ScalarLike:
        return (self.conj() * other).data.real

    def outer(self, other) -> types.ScalarLike:
        return (self.conj() * other).data.imag

    def in_nether(self) -> types.Self:
        return self.__class__.from_rect(self.x // 8, self.z // 8)

    def in_ring(self, ring_num: int | types.Iterable[int]) -> bool:
        """Checks whether coordinates are in the n-th stronghold ring."""

        a, b = inner_radii[ring_num], outer_radii[ring_num]
        return gm.in_interval(self.r, a, b)

    def closest_ring(self) -> int | types.Iterable[int]:
        """Finds the stronghold ring that is closest to the given coordinates."""

        radii = np.array(self.r)
        distances = np.abs(radii[..., None] - ring_radii)
        return distances.argmin(axis=-1) // 2
