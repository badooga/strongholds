from math import fsum

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from . import chunk_math as cm, generate as gen, locate as loc, math as gm, types

__all__ = ["Predict"]

# TODO: np.digitize? (https://stackoverflow.com/a/75727034)
# alternatively: use np.histogramdd, and set bin number for r axis
# to be the same as the number of r values
# then take transpose to flip the axes properly
# either way, **keep relevant_strongholds** filter
@np.vectorize(excluded=[2], signature="(),(n,m)->(62,60)")
def stronghold_histograms(p: types.Point, strongholds: types.Coordinates, bins: int = 60):
    """Creates a 2D histogram of the closest strongholds to p."""

    # filter out strongholds that aren't in a neighboring ring
    player_ring = cm.closest_ring(p)
    stronghold_rings = cm.closest_ring(strongholds)
    relevant_strongholds = strongholds[gm.in_interval(stronghold_rings,
                                                      player_ring - 1,
                                                      player_ring + 1)]

    # create 2D histogram
    closest_strongholds = np.array([loc.closest_stronghold(p, relevant_strongholds)])
    H, x_edges, z_edges = np.histogram2d(closest_strongholds.real,
                                         closest_strongholds.imag,
                                         bins=bins, density=False)

    x_centers, z_centers = gm.bin_centers(x_edges), gm.bin_centers(z_edges)

    # return counts and bin centers relative to p
    return np.vstack((H.T, x_centers - p.real, z_centers - p.imag))

#def _stronghold_histograms(p: types.Point)

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

        self.interpolator = self._create_interpolator(max_distance)

    def _create_interpolator(self, r_max: types.Scalar, bins: int = 60) -> NearestNDInterpolator:
        """
        Creates a `NearestNDInterpolator` for the probability that a given
        stronghold is the closest one to the player."""

        # create linspace of player distances from origin
        r = np.linspace(0, r_max, int(r_max/2))

        # compute histograms and separate relative x and z from vstack
        out = stronghold_histograms(r, self.heatmap, bins)
        counts, x, z = out[:, :bins, :], out[:, bins, :], out[:, bins+1, :]

        # effectively create a 3D meshgrid of r, x, z
        coords = np.zeros(shape=(*counts.shape, 3))
        i, j, k = np.indices(counts.shape)
        o = np.zeros_like(counts)

        coords[i, j, k, 0] = (r[:, None, None] + o)[i, j, k]
        coords[i, j, k, 1] = (x[:, :, None] + o)[i, j, k]
        coords[i, j, k, 2] = (z[:, None, :] + o)[i, j, k]

        return NearestNDInterpolator(coords.reshape(-1, 3), counts.ravel())

    @staticmethod
    def normalize_probabilities(probpoints: types.PointProbs) -> types.PointProbs:
        """Normalizes the probabilities in a dictionary of point probabilities."""

        total = fsum(probpoints.values())
        if not total:
            return probpoints
        return {point: p/total for point, p in probpoints.items()}

    def probability(self, player: types.Point, strongholds: types.Coordinates) -> types.PointProbs:
        """
        Finds the probability that the given strongholds will be the nearest one to the player.
        """

        r0, phi = gm.radius(player), gm.angle(player)

        # relative coordinates
        rel_coords = strongholds/gm.cis(phi) - r0

        # compute and normalize probabilities
        P = self.interpolator(r0, rel_coords.real, rel_coords.imag)
        P /= fsum(P)

        return dict(zip(strongholds, P))

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
        P = self.probability(player, new_targets)
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
