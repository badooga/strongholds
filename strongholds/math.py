import numpy as np

from .types import Coordinates, CoordinateTuples, NSequence, ScalarLike

# Create and manage points

def cis(phi: ScalarLike) -> ScalarLike:
    """Computes the complex exponential of phi."""

    return np.exp(1j * phi)

def rectpoint(x: ScalarLike, z: ScalarLike) -> Coordinates:
    """Creates Coordinates from x and z values."""

    return x + z*1j

def polarpoint(r: ScalarLike, phi: ScalarLike) -> Coordinates:
    """
    Creates Coordinates from r and phi values.
    """

    return r * cis(phi)

def to_xz(p: Coordinates) -> CoordinateTuples:
    """Converts complex Coordinates x + iz to array form [x, z]."""

    return p.view("(2,)float")

# Eucldiean properties

def radius(p: Coordinates) -> ScalarLike:
    """Finds the radius of the given coordinates."""

    return np.abs(p)

def angle(p: Coordinates, deg: bool = False) -> ScalarLike:
    """Finds the polar angle of the given coordinates."""

    return np.angle(p, deg)

def distance(p1: Coordinates, p2: Coordinates) -> ScalarLike:
    """Finds the distance between two coordinates."""

    return radius(p2 - p1)

# Other math

pm = np.array([1, -1])

def in_interval(x: ScalarLike, a: ScalarLike, b: ScalarLike) -> bool:
    """
    Returns whether x is in the closed interval [m, M], where
    m = min(a, b) and M = max(a, b). If any of these values
    are arrays, the operation is performed element-wise.
    """

    a, b = np.min((a, b)), np.max((a, b))
    return (a <= x) & (x <= b)

def rotate(p: Coordinates, delta: ScalarLike,
           origin: Coordinates | None = 0) -> Coordinates:
    """
    Rotates a point by delta radians counterclockwise
    about some origin point (or (0, 0), if not specified).
    """

    return origin + cis(delta) * (p - origin)

def unity_angles(n: int) -> NSequence:
    """
    Returns the arguments of the roots of the n-th roots of unity,
    i.e. the angles 2*pi/n, 4*pi/n, ..., 2*pi.
    """

    i = np.arange(n)
    return 2*np.pi*i/n

def bin_centers(bin_edges: NSequence) -> NSequence:
    """Converts histogram bin edges to bin centers."""

    return (bin_edges[1:] + bin_edges[:1])/2
