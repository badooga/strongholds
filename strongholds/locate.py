from . import math as gm

from .types import Point, Coordinates, ScalarLike

__all__ = ["closest_stronghold", "points_in_cone"]

def closest_stronghold(p: Point, strongholds: Coordinates) -> Point:
    """Finds the closest stronghold to a point."""

    distances = gm.distance(p, strongholds)
    return strongholds[distances.argmin(axis=-1, keepdims=True)].squeeze()

def points_in_cone(p: Point, grid: Coordinates,
                   theta: ScalarLike, theta_err: ScalarLike = 0.05,
                   error_is_relative: bool = False) -> Coordinates:
    """
    Finds the possible stronghold locations from an Eye of Ender throw.
    
    Keyword arguments:
        `p`: the location of the player
        `grid`: the possible stronghold grid points to consider (see `generation_grid`)
        `theta`: the angle of the Eye of Ender throw
        `theta_err`: the error of the throw
        `error_is_relative`: whether `theta_err` is absolute or relative
    """

    grid_rel = grid - p
    r, phi = gm.radius(grid_rel), gm.angle(grid_rel)
    if error_is_relative:
        theta_b, theta_a = theta * (1 + gm.pm * theta_err)
    else:
        theta_b, theta_a = theta + gm.pm * theta_err

    targets = (r > 0) & gm.in_interval(phi, theta_a, theta_b)

    return grid[targets]
