import numpy as np

from . import types

# Create and manage points

def cis(phi: types.ScalarLike) -> types.ScalarLike:
    """Computes the complex exponential of phi."""

    return np.exp(1j * phi)

def rectpoint(x: types.ScalarLike, z: types.ScalarLike) -> types.Coordinates:
    """Creates Coordinates from x and z values."""

    return x + z*1j

def polarpoint(r: types.ScalarLike, phi: types.ScalarLike) -> types.Coordinates:
    """
    Creates Coordinates from r and phi values.
    """

    return r * cis(phi)

def to_xz(p: types.Coordinates) -> types.CoordinateTuples:
    """Converts complex types.Coordinates x + iz to array form [x, z]."""

    return np.array([[p.real], [p.imag]]).T.squeeze()

# Eucldiean properties

def radius(p: types.Coordinates) -> types.ScalarLike:
    """Finds the radius of the given coordinates."""

    return np.abs(p)

def angle(p: types.Coordinates, deg: bool = False) -> types.ScalarLike:
    """Finds the polar angle of the given coordinates."""

    return np.angle(p, deg)

def distance(p1: types.Coordinates, p2: types.Coordinates) -> types.ScalarLike:
    """Finds the distance between two coordinates."""

    return radius(p2 - p1)

# Other math

pm = np.array([1, -1])

def in_interval(x: types.ScalarLike, a: types.ScalarLike, b: types.ScalarLike) -> bool:
    """
    Returns whether x is in the closed interval [m, M], where
    m = min(a, b) and M = max(a, b). If any of these values
    are arrays, the operation is performed element-wise.
    """

    a, b = np.min((a, b)), np.max((a, b))
    return (a <= x) & (x <= b)

def rotate(p: types.Coordinates, delta: types.ScalarLike,
           origin: types.Coordinates | None = 0) -> types.Coordinates:
    """
    Rotates a point by delta radians counterclockwise
    about some origin point (defaulting to the origin).
    """

    return origin + cis(delta) * (p - origin)

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
