import numpy as np

from .chunk_math import inner_radii, outer_radii, stronghold_count
from .generate import default_rng, generate_rings
from .locate import closest_stronghold
from .types import Coordinates, Generator, Iterable, NSequence, Point, ScalarLike

__all__ = ["generation_heatmap", "radial_pdf"]

def generation_heatmap(num_samples: int = 10**6,
                        ring_nums: Iterable[int] | None = None,
                        rng: Generator = default_rng,
                        concatenate: bool = True) -> NSequence | Iterable[Coordinates]:

    """
    For the supplied ring numbers, generates those rings the supplied number of times and return the result.
    """

    if ring_nums is None:
        ring_nums = range(8)

    stronghold_samples = np.array([
        generate_rings(ring_nums, True, rng) for _ in range(num_samples)
    ])

    if concatenate:
        stronghold_samples = np.concatenate(stronghold_samples)

    return stronghold_samples

def closest_stronghold_heatmap(p: Point, stronghold_sets: Iterable[Coordinates]) -> Coordinates:
    """
    Given an array from `generation_heatmap()` with `concatenate=False`,
    returns the stronghold in each set that is the closest to the given point.
    """

    return np.array([closest_stronghold(p, strongholds) for strongholds in stronghold_sets])


def radial_pdf(r: ScalarLike, ring_num: int) -> ScalarLike:
    """Computes the appproximate radial pdf for a given stronghold ring."""

    a, b = inner_radii[ring_num], outer_radii[ring_num]
    total = stronghold_count[ring_num]
    return r * 1/total # TODO
