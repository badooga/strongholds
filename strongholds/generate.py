import numpy as np

from . import chunk_math as cm, math as gm, types

default_rng = np.random.default_rng()

__all__ = ["generation_grid", "generate_ring", "generate_rings", "generate_all", "generation_heatmap"]

def generation_grid(ring_nums: types.Iterable | None = None,
                    center: bool = False) -> cm.Coordinates:
    """Returns a grid of possible stronghold points in the supplied rings."""

    if ring_nums is None:
        ring_nums = range(8)

    i = np.arange(-24496/16, 24496/16)
    X, Z = np.meshgrid(16*i, 16*i)
    if center:
        X += 8
        Z += 8
    grid = cm.Coordinates.from_rect(X.flatten(), Z.flatten())

    rings = np.any([grid.in_ring(n) for n in ring_nums], axis=0)
    return grid[rings]

def generate_ring(ring_num: int, snap: bool = True,
                  rng: types.Generator = default_rng) -> cm.Coordinates:
    """Generates stronghold coordinates in a given ring."""

    n = cm.stronghold_count[ring_num]
    a, b = cm.inner_radii[ring_num], cm.outer_radii[ring_num]

    r = rng.uniform(a, b, n)
    phi = rng.uniform(0, 2*np.pi) + gm.unity_angles(n)
    P = cm.Coordinates.from_polar(r, phi)

    if snap:
        # first, snaps biome to the nearest chunk origin
        P = P.chunk_corner
        # next, snaps to uniformly chosen biome center up to 7 chunks away
        biome_snap = 16 * rng.integers(-7, 8, (2, n))
        P.coords += biome_snap[0] + 1j * biome_snap[1]

    return P

def generate_rings(ring_nums: types.Iterable, snap: bool = True,
                   rng: types.Generator = default_rng) -> cm.Coordinates:
    """Generates stronghold coordinates in given rings."""

    return cm.Coordinates(np.concatenate([generate_ring(n, snap, rng).coords
                                          for n in ring_nums]))

def generate_all(snap: bool = True, rng: types.Generator = default_rng) -> cm.Coordinates:
    """Generates all 128 random strongholds a world can have."""

    return generate_rings(range(8), snap, rng)

def generation_heatmap(num_samples: int = 10**6,
                        ring_nums: types.Iterable[int] | None = None,
                        rng: types.Generator = default_rng,
                        snap: bool = True, concatenate: bool = True
                        ) -> cm.Coordinates | types.CoordinateSets:

    """
    For the supplied ring numbers, generates those rings
    the supplied number of times and return the result.
    """

    if ring_nums is None:
        ring_nums = range(8)

    stronghold_samples = np.array([
        generate_rings(ring_nums, snap, rng).coords for _ in range(num_samples)
    ])

    if concatenate:
        stronghold_samples = cm.Coordinates(np.concatenate(stronghold_samples))

    return stronghold_samples
