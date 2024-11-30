from collections import UserDict

from math import fsum

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import chunk_math as cm, generate as gen, graphing, locate as loc, math as gm, types

__all__ = ["Predict"]

class Probabilities(UserDict[types.Point, types.Scalar]):
    """Class for storing stronghold probabilities."""

    @classmethod
    def from_arrays(cls, points: types.Points, probabilities: types.NSequence) -> types.Self:
        self = cls(dict(zip(points, probabilities)))
        self.normalize()
        return self

    @property
    def points(self) -> types.Points:
        return np.array(list(self.keys()))

    @property
    def probabilities(self) -> types.NSequence:
        return np.array(list(self.values()))

    def remove_zeros(self) -> None:
        for k, v in self.copy().items():
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

    def view(self, threshold: types.Scalar = 0):
        items = [(cm.Coordinates(k), v) for k, v in self.items() if v >= threshold]
        return sorted(items, key=lambda i: i[1], reverse=True)

class Predict:
    """Class for predicting where the closest stronghold will be."""

    def __init__(self, grid: cm.Coordinates | None = None,
                 heatmap: types.CoordinateSets | None = None,
                 rng: types.Generator = gen.default_rng) -> None:

        if grid is None:
            grid = gen.generation_grid()
        self.grid = grid

        if heatmap is None:
            heatmap = gen.generation_heatmap(10**6, rng=rng, concatenate=False)
        self.heatmap = heatmap

        self.throws: list[loc.EyeThrow] = []
        self.individual_probs: list[Probabilities] = []
        self.cumulative_probs: Probabilities = []

        self.interpolators: list[RegularGridInterpolator] = []

    def create_interpolator(self, player: cm.Coordinates, bins: int = 60) -> RegularGridInterpolator:
        """Creates an interpolator for the nearest strongholds to a point."""

        # bin the coordinates
        closest_strongholds = loc.closest_stronghold(player, self.heatmap)
        H, x_edges, z_edges = np.histogram2d(closest_strongholds.x,
                                            closest_strongholds.z,
                                            bins=bins, density=False)

        # interpolate the result
        x_centers, z_centers = gm.bin_centers(x_edges), gm.bin_centers(z_edges)
        return RegularGridInterpolator((x_centers, z_centers), H,
                                       bounds_error=False, fill_value=0)

    def find_probabilities(self, player: cm.Coordinates, strongholds: cm.Coordinates) -> Probabilities:
        """
        Finds the probabilities that the given strongholds will be the nearest one to the player.
        """

        interpolator = self.create_interpolator(player)
        self.interpolators.append(interpolator)
        P = interpolator(strongholds.to_xz())

        return Probabilities.from_arrays(strongholds.coords, P)

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
        players = np.array([throw.location.coords for throw in self.throws])
        scatter_players = ax.scatter(players.real, players.imag, marker="x", color="red")

        scatter_grid = ax.scatter(self.grid.x, self.grid.z,
                                  s=1e-4, color="white")

        t = np.linspace(0, np.abs(self.heatmap).max())
        plot_rays_a = []
        plot_rays_b = []
        for throw in self.throws:
            ray_a = throw.location.coords + throw.ray_a.coords * t
            ray_b = throw.location.coords + throw.ray_b.coords * t
            plot_rays_a += ax.plot(ray_a.real, ray_a.imag, lw=0.375, ls="--", color="orange")
            plot_rays_b += ax.plot(ray_b.real, ray_b.imag, lw=0.375, ls="--", color="orange")

        scatter_intersection = ax.scatter(self.cumulative_probs.points.real,
                                          self.cumulative_probs.points.imag,
                                          s=100*self.cumulative_probs.probabilities,
                                          color="green")

        graphing.flip_zaxis(ax)

        #return (scatter_players, plot_rays_0, plot_rays_a,
        #        plot_rays_b, scatter_grid, scatter_intersection)
        return (scatter_players, plot_rays_a, plot_rays_b,
                scatter_grid, scatter_intersection)
