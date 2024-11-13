from math import fsum

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from . import chunk_math as cm, generate as gen, locate as loc, math as gm, types

__all__ = ["Predict"]

@np.vectorize(excluded=[2], signature="(),(n,m)->(62,60)")
def stronghold_histograms(p: types.Point, strongholds: types.Coordinates, bins: int = 60):
    """Creates a 2D histogram of the closest strongholds to p."""

    closest_strongholds = loc.closest_stronghold(p, strongholds)
    H, x_edges, z_edges = np.histogram2d(closest_strongholds.real,
                                         closest_strongholds.imag,
                                         bins=bins, density=False)

    x_centers, z_centers = gm.bin_centers(x_edges), gm.bin_centers(z_edges)

    return np.vstack((H.T, x_centers - p.real, z_centers - p.imag))

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

        self.throw_probabilities: types.PointProbs = {}
        self.stronghold_probabilities: list[types.PointProbs] = []

        if max_distance is None:
            max_distance = gm.radius(self.grid).max()

        self.interpolator = self._create_interpolator(max_distance)

    def _create_interpolator(self, r_max: types.Scalar, bins: int = 60) -> NearestNDInterpolator:
        """docstring"""

        #r0_max = gm.radius(self.grid).max()

        r = np.linspace(0, r_max, int(r_max/2))

        out = stronghold_histograms(r, self.heatmap, bins)

        counts, x, z = out[:, :bins, :], out[:, bins, :], out[:, bins+1, :]

        coords = np.zeros(shape=(*counts.shape, 3))
        i, j, k = np.indices(counts.shape)
        o = np.zeros_like(counts)

        coords[i, j, k, 0] = (r[:, None, None] + o)[i, j, k]
        coords[i, j, k, 1] = (x[:, :, None] + o)[i, j, k]
        coords[i, j, k, 2] = (z[:, None, :] + o)[i, j, k]

        return NearestNDInterpolator(coords.reshape(-1, 3), counts.ravel())

    @staticmethod
    def normalize_probabilities(probabilities: types.PointProbs) -> types.PointProbs:
        """Normalizes the probabilities in a dictionary of point probabilities."""

        total = fsum(P := probabilities.values())
        return dict(zip(probabilities, map(lambda x: x/total, P)))

    def points_in_cone(self, player: types.Point,
                       angle: types.Scalar,
                       angle_error: types.Scalar) -> types.Coordinates:
        """Finds the possible stronghold locations from an Eye of Ender throw."""

        angle = cm.to_radians(angle)
        angle_error *= np.pi/180

        return loc.points_in_cone(player, self.grid, angle, angle_error)

    def interpolate(self, strongholds: types.Iterable[types.Point]):
        """interpolate"""

        # interpolate
        P = 1

        probabilities = dict(zip())
        return self.normalize_probabilities(probabilities)

    def add_throw(self, player: types.Point,
                  angle: types.Scalar,
                  angle_error: types.Scalar = 0.1) -> None:
        """docstring"""

        possible_grid_points = self.points_in_cone(player, angle, angle_error)
        old_targets = self.stronghold_probabilities.keys()

        new_targets = old_targets & possible_grid_points

        new_probabilities = dict(zip(new_targets, self.interpolate(new_targets)))
