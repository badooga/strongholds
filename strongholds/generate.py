from typing import Iterable

import numpy as np

from . import chunk_math as cm
from . import geometry as gm

from .types import Generator, RectCoordinates

default_rng = np.random.default_rng(0)

def stronghold_grid(ring_nums: Iterable | None = None) -> RectCoordinates:
    """Returns a grid of possible stronghold points in the supplied rings."""

    if ring_nums is None:
        ring_nums = range(8)

    j = np.arange(-24496/16, 24496/16)
    X = Z = 16*j + 8

    grid = np.array(np.meshgrid(X, Z)).T.reshape(-1, 2)
    rings = np.any([cm.in_ring(grid, n) for n in ring_nums], axis=0)
    return grid[rings]

def generate_ring(ring_num: int, snap: bool = True,
                  rng: Generator = default_rng) -> RectCoordinates:
    """Generates stronghold coordinates in a given ring."""

    n = cm.stronghold_count[ring_num]
    a, b = cm.inner_radii[ring_num], cm.outer_radii[ring_num]

    phi = rng.uniform(0, 2*np.pi) + gm.unity_angles(n)
    r = rng.uniform(a, b, n)
    P = gm.to_rect(gm.polarpoint(r, phi))
    if snap:
        P = cm.snap_chunk(P) + 16 * rng.integers(-7, 8, (n, 2)) + 8
    return P

def generate_strongholds(snap: bool = True, rng: Generator = default_rng) -> RectCoordinates:
    """Generates the 128 random strongholds a world can have."""

    return np.concatenate([generate_ring(n, snap, rng) for n in range(8)])
