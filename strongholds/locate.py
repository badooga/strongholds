from dataclasses import dataclass

from . import chunk_math as cm, math as gm, types

__all__ = ["closest_stronghold", "EyeThrow"]

def closest_stronghold(p: types.Coordinates,
                       s: types.Coordinates | types.CoordinateSets) -> types.Coordinates:
    """
    Finds the closest stronghold from coordinates `s` to the player `p`.

    Broadcasting rules:
    - If `p` is an array of player locations, it will do this for each one
    - If `s` is of type `CoordinateSets`, i.e. an array of stronghold
      coordinate arrays each representing a different world, it will
      do this for each world.
    - In general, the output shape is `p.shape + s.shape[:-1]`.
    """

    # adds empty axes to so that we can do broadcasting for |s - p|
    p = gm.np.array(p)
    N = (None for _ in range(s.ndim))
    P = p[..., *N]

    # finds the mins along the last axis
    m = gm.distance(s, P).min(axis=-1, keepdims=True)

    # tiles s to have the same shape as m
    S = gm.np.tile(s, (*p.shape, *gm.np.ones_like(s.shape)))

    # identifies the strongholds in s that minimize |s - p|
    M = gm.np.where(gm.distance(S, P) == m, S, 0)
    return M.sum(axis=-1)
@dataclass
class EyeThrow:
    """Stores the data of an Eye of Ender throw."""

    location: types.Point
    angle: types.Scalar
    error: types.Scalar

    def __post_init__(self) -> None:
        # store rectangular and polar coordinates
        self.x: types.Scalar = self.location.real
        self.z: types.Scalar = self.location.imag
        self.r: types.Scalar = gm.radius(self.location)
        self.phi: types.Scalar = gm.angle(self.location)

        # convert throw angle to radians
        self.theta: types.Scalar = cm.to_radians(self.angle)
        self.dtheta: types.Scalar = self.error * gm.np.pi/180

        # store throw angle error interval
        self.theta_a: types.Scalar = self.theta - self.dtheta
        self.theta_b: types.Scalar = self.theta + self.dtheta

    def points_in_cone(self, grid: types.Coordinates) -> types.Coordinates:
        """Finds the possible grid locations the throw could be pointing towards."""

        grid_phi = gm.angle(grid - self.location)
        return grid[gm.in_interval(grid_phi, self.theta_a, self.theta_b)]
