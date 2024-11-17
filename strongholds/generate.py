import numpy as np

from . import chunk_math as cm, math as gm, types

default_rng = np.random.default_rng()

__all__ = ["generation_grid", "generate_ring", "generate_rings", "generate_all", "generation_heatmap"]

def generation_grid(ring_nums: types.Iterable | None = None, center: bool = False) -> types.Coordinates:
    """Returns a grid of possible stronghold points in the supplied rings."""

    if ring_nums is None:
        ring_nums = range(8)

    i = np.arange(-24496/16, 24496/16)
    x = z = 16*i + (8 if center else 0)
    X, Z = np.meshgrid(x, z)
    grid = gm.rectpoint(X, Z).flatten()

    rings = np.any([cm.in_ring(grid, n) for n in ring_nums], axis=0)
    return grid[rings]

def generate_ring(ring_num: int, snap: bool = True,
                  rng: types.Generator = default_rng,
                  center: bool = False) -> types.Coordinates:
    """Generates stronghold coordinates in a given ring."""

    n = cm.stronghold_count[ring_num]
    a, b = cm.inner_radii[ring_num], cm.outer_radii[ring_num]

    r = rng.uniform(a, b, n)
    phi = rng.uniform(0, 2*np.pi) + gm.unity_angles(n)
    P = gm.polarpoint(r, phi)

    if snap:
        # first, snaps biome to the nearest chunk origin
        P = cm.snap_chunk(P)
        # next, snaps to uniformly chosen biome center up to 7 chunks away
        biome_snap = 16 * rng.integers(-7, 8, (2, n))
        biome_snap += 8 if center else 0
        P += biome_snap[0] + 1j * biome_snap[1]
    return P

def generate_rings(ring_nums: types.Iterable, snap: bool = True,
                   rng: types.Generator = default_rng) -> types.Coordinates:
    """Generates stronghold coordinates in given rings."""

    return np.concatenate([generate_ring(n, snap, rng) for n in ring_nums])

def generate_all(snap: bool = True, rng: types.Generator = default_rng) -> types.Coordinates:
    """Generates all 128 random strongholds a world can have."""

    return generate_rings(range(8), snap, rng)

def generation_heatmap(num_samples: int = 10**6,
                        ring_nums: types.Iterable[int] | None = None,
                        rng: types.Generator = default_rng,
                        snap: bool = True, concatenate: bool = True
                        ) -> types.Coordinates | types.CoordinateSets:

    """
    For the supplied ring numbers, generates those rings
    the supplied number of times and return the result.
    """

    if ring_nums is None:
        ring_nums = range(8)

    stronghold_samples = np.array([
        generate_rings(ring_nums, snap, rng) for _ in range(num_samples)
    ])

    if concatenate:
        stronghold_samples = np.concatenate(stronghold_samples)

    return stronghold_samples
