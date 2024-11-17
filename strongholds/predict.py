from collections import UserDict

from math import fsum

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import chunk_math as cm, generate as gen, graphing, locate as loc, math as gm, types

__all__ = ["Predict"]

class Probabilities(UserDict[types.Point, types.Scalar]):
    """Class for storing stronghold probabilities."""

    @classmethod
    def from_arrays(cls, points: types.Coordinates, probabilities: types.NSequence) -> types.Self:
        self = cls(dict(zip(points, probabilities)))
        self.normalize()
        return self

    @property
    def points(self) -> types.Coordinates:
        return np.array(list(self.keys()))

    @property
    def probabilities(self) -> types.NSequence:
        return np.array(list(self.values()))

    def remove_zeros(self) -> None:
        for k, v in self.items():
            if not v:
                del self[k]

    def normalize(self) -> None:
        self.remove_zeros()
        total = self.probabilities.sum()
        if total:
            for k in self:
                self[k] /= total

    def __and__(self, other: types.Self) -> None:
        self.data = {k: self[k] * other[k] for k in self.keys() & other.keys()}

    def intersection(self, *args: types.Self) -> None:
        for other in args:
            self & other
            self.remove_zeros()
        self.normalize()

class Predict:

    def __init__(self, grid: types.Coordinates | None = None,
                 heatmap: types.CoordinateSets | None = None,
                 max_distance: types.Scalar | None = None,
                 rng: types.Generator = gen.default_rng) -> None:
        """docstring"""

        if grid is None:
            grid = gen.generation_grid()
        self.grid = grid

        if heatmap is None:
            heatmap = gen.generation_heatmap(10**6, rng=rng, concatenate=False)
        self.heatmap = heatmap

        self.throws: list[loc.EyeThrow] = []
        self.individual_probs: list[Probabilities] = []
        self.cumulative_probs: Probabilities = []

        if max_distance is None:
            max_distance = gm.radius(self.heatmap).max()

    def create_interpolator(self, player: types.Point, bins: int = 60) -> RegularGridInterpolator:
        """Creates an interpolator for the nearest strongholds to a point."""

        # bin the coordinates
        closest_strongholds = loc.closest_stronghold(player, self.heatmap)
        H, x_edges, z_edges = np.histogram2d(closest_strongholds.real,
                                            closest_strongholds.imag,
                                            bins=bins, density=False)

        # interpolate the result
        x_centers, z_centers = gm.bin_centers(x_edges), gm.bin_centers(z_edges)
        return RegularGridInterpolator((x_centers, z_centers), H.T,
                                       bounds_error=False, fill_value=0)

    def find_probabilities(self, player: types.Point, strongholds: types.Coordinates) -> Probabilities:
        """
        Finds the probabilities that the given strongholds will be the nearest one to the player.
        """

        interpolator = self.create_interpolator(player)
        P = interpolator(gm.to_xz(strongholds))

        return Probabilities.from_arrays(strongholds, P)

    def add_throw(self, player: types.Point,
                  angle: types.Scalar,
                  angle_error: types.Scalar = 0.1) -> None:
        """
        Adds an Eye of Ender throw to the list of throws and computes the resulting probabilities.
        """

        # saves the throw data
        throw = loc.EyeThrow(player, angle, angle_error)
        self.throws.append(throw)

        # finds the possible grid points for this throw
        new_targets = throw.points_in_cone(self.grid)

        # compute the probabilities for each grid point from just this throw
        new_probs = self.find_probabilities(player, new_targets)

        # find their intersection with previous throw grid points, if any
        # if there are old throws, multiply those probabilities and normalize
        if self.cumulative_probs:
            new_probs.intersection(*self.individual_probs)

        self.cumulative_probs = new_probs
        self.individual_probs.append(new_probs)

    def plot_throws(self, fig: graphing.Figure, ax: graphing.Axes):
        pass
        # TODO: graphing.flip_zaxis(ax)
