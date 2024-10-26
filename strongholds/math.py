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
    """Creates Coordinates from r and phi values."""

    return r * cis(phi)

def to_xz(p: Coordinates) -> CoordinateTuples:
    """Converts complex Coordinates x + iz to array form [x, z]."""

    return p.view("(2,)float")

# Eucldiean properties

def radius(p: Coordinates) -> ScalarLike:
    """Finds the radius of given coordinates."""

    return np.abs(p)

def angle(p: Coordinates) -> ScalarLike:
    """Finds the polar angle of given coordinates."""

    return np.angle(p)

def distance(p1: Coordinates, p2: Coordinates) -> ScalarLike:
    """Euclidean distance metric between two coordinates."""

    return radius(p2 - p1)

# Other math

pm = np.array([1, -1])

def in_interval(x: ScalarLike, a: ScalarLike, b: ScalarLike) -> bool:
    """Returns whether x is in the closed interval [a, b]."""

    a, b = np.min((a, b)), np.max((a, b))
    return (a <= x) & (x <= b)

def rotate(p: Coordinates, delta: ScalarLike,
           origin: Coordinates | None = 0) -> Coordinates:
    """Rotates a point by delta radians about some origin point."""

    return origin + cis(delta) * (p - origin)

def unity_angles(n: int) -> NSequence:
    """Returns the arguments of the roots of the n-th roots of unity."""    

    i = np.arange(n)
    return 2*np.pi*i/n
