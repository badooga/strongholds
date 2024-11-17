from math import fsum

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import chunk_math as cm, generate as gen, locate as loc, math as gm, types

__all__ = ["Predict"]

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

        self.throw_locations: list[types.Point] = []
        self.throw_angles: list[types.Scalar] = []
        self.throw_errors: list[types.Scalar] = []

        self.individual_throws: list[types.PointProbs] = []
        self.cumulative_throw: types.PointProbs = {}

        if max_distance is None:
            max_distance = gm.radius(self.heatmap).max()

    def create_interpolator(self, player: types.Point, bins: int = 60) -> RegularGridInterpolator:
        """docstring"""

        # isolate strongholds in specific rings
        n = cm.closest_ring(player)
        relevant_strongholds = self.heatmap[..., max(n, 0):n+1]

        # bin the coordinates
        closest_strongholds = np.array([loc.closest_stronghold(player, relevant_strongholds)])
        H, x_edges, z_edges = np.histogram2d(closest_strongholds.real,
                                            closest_strongholds.imag,
                                            bins=bins, density=True)

        # interpolate the result
        x_centers, z_centers = gm.bin_centers(x_edges), gm.bin_centers(z_edges)
        return RegularGridInterpolator((x_centers, z_centers), H.T,
                                       bounds_error=False, fill_value=True)

    @staticmethod
    def normalize_probabilities(probpoints: types.PointProbs) -> types.PointProbs:
        """Normalizes the probabilities in a dictionary of point probabilities."""

        total = fsum(probpoints.values())
        if not total:
            return probpoints
        return {point: p/total for point, p in probpoints.items()}

    def find_probabilities(self, player: types.Point, strongholds: types.Coordinates) -> types.PointProbs:
        """
        Finds the probabilities that the given strongholds will be the nearest one to the player.
        """

        interpolator = self.create_interpolator(player)
        P = interpolator(gm.to_xz(strongholds))

        return self.normalize_probabilities(dict(zip(strongholds, P)))

    def points_in_cone(self, player: types.Point,
                       angle: types.Scalar,
                       angle_error: types.Scalar) -> types.Coordinates:
        """Finds the possible stronghold locations from an Eye of Ender throw."""

        angle = cm.to_radians(angle)
        angle_error *= np.pi/180

        return loc.points_in_cone(player, self.grid, angle, angle_error)

    def add_throw(self, player: types.Point,
                  angle: types.Scalar,
                  angle_error: types.Scalar = 0.1) -> None:
        """
        Adds an Eye of Ender throw to the list of throws and computes the resulting probabilities.
        """

        # find the possible grid points from this throw
        new_targets = self.points_in_cone(player, angle, angle_error)

        # find their intersection with previous throw grid points, if any
        old_targets = self.cumulative_throw.keys()
        if old_targets:
            new_targets = np.fromiter(old_targets & new_targets, complex)

        # compute the probabilities for each grid point from just this throw
        P = self.find_probabilities(player, new_targets)
        new_throw = dict(zip(new_targets, P))
        self.individual_throws.append(new_throw)

        # if there are no other throws, we're done
        if not old_targets:
            self.cumulative_throw = new_throw
            return

        # if there are other throws, multiply those probabilities and normalize
        intersection = [self.normalize_probabilities({k: v for k, v in throw.items()
                                                       if k in new_throw})
                         for throw in self.individual_throws]

        self.cumulative_throw = self.normalize_probabilities({
            k: np.prod([throw[k] for throw in intersection])
            for k in new_throw})
