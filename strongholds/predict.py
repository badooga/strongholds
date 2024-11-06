import numpy as np
from math import fsum
from scipy.interpolate import LinearNDInterpolator

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

        self.throw_probabilities: types.PointProbs = {}
        self.stronghold_probabilities: list[types.PointProbs] = []

        if max_distance is None:
            max_distance = gm.radius(self.grid).max()

        self.interpolator = self._create_interpolator(max_distance)

    def _create_interpolator(self, r0_max: types.Scalar):
        """docstring"""

        #r0_max = gm.radius(self.grid).max()

        x, z = [], []
        r0 = np.linspace(0, r0_max, int(r0_max/2))

        counts = []

        for r0_ in r0:
            closest_strongholds = loc.closest_stronghold(r0_, self.heatmap)

            H, x_edges, z_edges = np.histogram2d(closest_strongholds.real,
                                                    closest_strongholds.imag,
                                                    bins=60, density=True)
            x_centers, z_centers = gm.bin_centers(x_edges), gm.bin_centers(z_edges)

            x.append(x_centers - r0_) # relative distance
            z.append(z_centers)
            counts.append(H.T)

        x, z, r0 = np.array(x), np.array(z), np.array(r0)
        X, Z, R0 = np.meshgrid(x, z, r0)
        points = np.dstack((X.ravel(), Z.ravel(), R0.ravel()))

        counts = np.array(counts)
        values = counts.ravel()

        return LinearNDInterpolator(points, values, fill_value=0, rescale=True)

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
