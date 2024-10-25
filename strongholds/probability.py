import numpy as np

from .generate import default_rng, generate_rings, stronghold_grid
from .types import Iterable, Generator, NVector

def probability_heatmap(num_samples: int = 10**6,
                        ring_nums: Iterable[int] | None = None,
                        rng: Generator = default_rng) -> NVector:

    """
    For the supplied ring numbers, generates that ring many times and return the result.
    """

    if ring_nums is None:
        ring_nums = range(8)

    stronghold_samples = np.concatenate([
        generate_rings(ring_nums, True, rng) for _ in range(num_samples)
    ])

    return stronghold_samples
