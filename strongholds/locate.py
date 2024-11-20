from dataclasses import dataclass

from . import chunk_math as cm, math as gm, types

__all__ = ["closest_stronghold", "EyeThrow"]

def closest_stronghold(p: cm.Coordinates,
                       s: cm.Coordinates | types.CoordinateSets) -> cm.Coordinates:
    """
    Finds the closest stronghold from coordinates `s` to the player `p`.

    Broadcasting rules:
    - If `p` is an array of player locations, it will do this for each one
    - If `s` is of type `CoordinateSets`, i.e. an array of stronghold
      coordinate arrays each representing a different world, it will
      do this for each world.
    - In general, the output shape is `p.shape + s.shape[:-1]`.
    """

    try:
        p = p.coords
        s = s.coords
    except AttributeError:
        pass

    # adds empty axes to so that we can do broadcasting for |s - p|
    p = gm.np.array(p)
    N = (None for _ in range(s.ndim))
    P = p[..., *N]

    # finds the mins along the last axis
    m = gm.np.abs(s - P).min(axis=-1, keepdims=True)

    # tiles s to have the same shape as m
    S = gm.np.tile(s, (*p.shape, *gm.np.ones_like(s.shape)))

    # identifies the strongholds in s that minimize |s - p|
    M = gm.np.where(gm.np.abs(S - P) == m, S, 0)
    return cm.Coordinates(M.sum(axis=-1))

@dataclass
class EyeThrow:
    """Stores the data of an Eye of Ender throw."""

    location: cm.Coordinates
    angle: types.Scalar
    error: types.Scalar

    def __post_init__(self) -> None:
        # convert throw angle to radians
        self.theta: types.Scalar = cm.to_radians(self.angle)
        self.dtheta: types.Scalar = self.error * gm.np.pi/180

        # store throw angle error interval
        self.theta_a: types.Scalar = self.theta - self.dtheta
        self.theta_b: types.Scalar = self.theta + self.dtheta

        # stores unit vectors for "throw cone"
        self.ray_a = cm.Coordinates.phasor(self.theta_a)
        self.ray_b = cm.Coordinates.phasor(self.theta_b)

    def points_in_cone(self, grid: cm.Coordinates) -> cm.Coordinates:
        """Finds the possible grid locations the throw could be pointing towards."""

        # shift the grid to the eye throw location as its origin
        grid_rel = grid.relative_to(self.location)

        # check if grid_rel is between ray_a and ray_b
        # see https://stackoverflow.com/a/17497339
        mask1 = (self.ray_a.outer(grid_rel) * self.ray_a.outer(self.ray_b)) >= 0
        mask2 = (self.ray_b.outer(grid_rel) * self.ray_b.outer(self.ray_a)) >= 0
        mask3 = gm.np.isclose(grid_rel.r, 0)

        # apply mask
        return cm.Coordinates(grid.coords[(mask1 & mask2) | mask3])
