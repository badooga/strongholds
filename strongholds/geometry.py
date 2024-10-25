import numpy as np

from .types import (Scalar, NVector, ScalarLike, RectCoordinates, PolarCoordinates)

# Create points

def rectpoint(x: ScalarLike, z: ScalarLike) -> RectCoordinates:
    """Creates RectCoordinates from x and z values."""

    return np.array([x, z]).T

def polarpoint(r: ScalarLike, phi: ScalarLike) -> PolarCoordinates:
    """Creates PolarCoordinates from r and phi values.."""

    return np.array([r, phi]).T

# Eucldiean properties

def distance(p1: RectCoordinates, p2: RectCoordinates) -> ScalarLike:
    """Euclidean distance metric between two coordinates."""

    d = p2 - p1
    return np.sqrt(np.einsum("...i->...", d**2))

def radius(p: RectCoordinates) -> ScalarLike:
    """Finds the radius of given coordinates."""

    return distance(p, 0)

def angle(p: RectCoordinates) -> ScalarLike:
    """Finds the polar angle of given coordinates."""

    return np.arctan2(*p.T)

# Convert between coordinate systems

def to_rect(p: PolarCoordinates) -> RectCoordinates:
    """Converts from polar coordinates to rectangular coordinates."""

    r, phi = p.T
    x, z = r * np.cos(phi), r * np.sin(phi)
    return rectpoint(x, z)

def to_polar(p: RectCoordinates) -> PolarCoordinates:
    """Converts from rectangular to polar coordinates."""

    r, phi = radius(p), angle(p)
    return polarpoint(r, phi)

# Other math

pm = np.array([1, -1])

def in_interval(x: ScalarLike, a: ScalarLike, b: ScalarLike) -> bool:
    """Returns whether x is in the closed interval [a, b]."""

    a, b = np.min((a, b)), np.max((a, b))
    return (a <= x) & (x <= b)

def rotate(p: RectCoordinates, theta: Scalar,
           origin: RectCoordinates | None = None) -> RectCoordinates:
    """Rotates a point by theta radians about some origin."""

    if origin is None:
        origin = rectpoint(0, 0)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.einsum("...jk,...k->...j", R, p - origin) + origin

def unity_angles(n: int) -> NVector:
    """Returns the arguments of the roots of the n-th roots of unity."""    

    j = np.arange(n)
    return 2*np.pi*j/n
