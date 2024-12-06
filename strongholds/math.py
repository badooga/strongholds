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

    return np.exp(-((y - mean)/std)**2/2)/np.sqrt(2*np.pi*std**2)
