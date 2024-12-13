import numpy as np

from . import types


def phasor(phi: types.ScalarLike, deg: bool = False) -> types.ScalarLike:
    """Computes the complex exponential exp(j*phi)."""

    phi *= np.pi/180 if deg else 1
    return np.exp(1j * phi)


def in_interval(x: types.ScalarLike, a: types.ScalarLike, b: types.ScalarLike) -> bool:
    """
    Returns whether x is in the closed interval [m, M], where
    m = min(a, b) and M = max(a, b). If any of these values
    are arrays, the operation is performed element-wise.
    """

    a, b = np.min((a, b)), np.max((a, b))
    return (a <= x) & (x <= b)


def unity_angles(n: int) -> types.NSequence:
    """
    Returns the arguments of the roots of the n-th roots of unity,
    i.e. the angles 2*pi/n, 4*pi/n, ..., 2*pi.
    """

    i = np.arange(n)
    return 2*np.pi*i/n


def bin_centers(bin_edges: types.NSequence) -> types.NSequence:
    """Converts histogram bin edges to bin centers."""

    return (bin_edges[1:] + bin_edges[:-1])/2


def normal(y: types.ScalarLike, mean: types.ScalarLike,
           std: types.ScalarLike) -> types.ScalarLike:
    """Computes the pdf for a normal distribution."""

    return np.exp(-((y - mean)/std)**2 / 2)/np.sqrt(2 * np.pi * std**2)


class Coordinates2D(np.ndarray):
    """Stores 2D points in complex form."""

    def __new__(cls, coords: types.Self | types.PointLike) -> types.Self:
        """Constructs the Coordinates from complex data.

        Args:
            coords (Self | PointLike): The coordinates in complex form.
        """

        obj = np.asarray(coords, dtype=np.complex128).view(cls)
        return obj

    def __repr__(self) -> str:
        return self.to_xz().__repr__()

    @classmethod
    def from_rect(cls, x: types.ScalarLike, z: types.ScalarLike) -> types.Self:
        """Constructs the Coordinates from [x, z] data.

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

        return cls(r * phasor(phi, deg=deg))

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

    def to_xz(self):
        return np.stack((self.x, self.z), -1)

    def rotated(self, delta: types.ScalarLike,
                origin: types.Self | None = None,
                deg: bool = False) -> types.Self:
        """Rotates the Coordinates by an angle delta about some origin point.

        Args:
            delta (ScalarLike): The counterclockwise angle to rotate by in the xz plane.
            origin (Self | None, optional): The point to rotate around. Defaults to None (the origin).
            deg (bool, optional): Whether to consider delta in degrees. Defaults to False.
        """

        if origin is None:
            origin = self.__class__(0)

        return origin + phasor(delta, deg=deg) * (self - origin)

    def relative_angle(self, other, direction: types.Self | None = None) -> types.ScalarLike:
        rel_phi = (self - other).phi
        if direction is not None:
            rel_phi -= direction.phi
        return rel_phi

    def inner(self, other) -> types.ScalarLike:
        return (self.conj() * other).data.real

    def outer(self, other) -> types.ScalarLike:
        return (self.conj() * other).data.imag
