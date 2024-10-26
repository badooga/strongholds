import numpy as np

from .chunk_math import inner_radii, outer_radii, stronghold_count
from .generate import default_rng, generate_rings
from .types import Generator, Iterable, NSequence, ScalarLike

__all__ = ["generation_heatmap", "radial_pdf"]

def generation_heatmap(num_samples: int = 10**6,
                        ring_nums: Iterable[int] | None = None,
                        rng: Generator = default_rng) -> NSequence:

    """
    For the supplied ring numbers, generates those rings the supplied number of times and return the result.
    """

    if ring_nums is None:
        ring_nums = range(8)

    stronghold_samples = np.concatenate([
        generate_rings(ring_nums, True, rng) for _ in range(num_samples)
    ])

    return stronghold_samples

def radial_pdf(r: ScalarLike, ring_num: int) -> ScalarLike:
    """Computes the appproximate radial pdf for a given stronghold ring."""

    a, b = inner_radii[ring_num], outer_radii[ring_num]
    total = stronghold_count[ring_num]
    return r * 1/total # TODO
