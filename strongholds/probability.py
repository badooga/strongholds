import numpy as np

from . import chunk_math as cm, generate as gen, locate as loc, types

from .generate import default_rng, generate_rings
from .locate import closest_stronghold
from .types import Coordinates, Generator, Iterable, CoordinateSets

from . import types

__all__ = ["generation_heatmap", "Predict"]

def generation_heatmap(num_samples: int = 10**6,
                        ring_nums: Iterable[int] | None = None,
                        rng: Generator = default_rng,
                        snap: bool = True,
                        concatenate: bool = True) -> Coordinates | CoordinateSets:

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
