from . import geometry as gm

from .types import RectPoint, RectCoordinates, ScalarLike

__all__ = ["closest_stronghold", "points_in_cone"]

def closest_stronghold(p: RectPoint, strongholds: RectCoordinates) -> RectPoint:
    """Finds the closest stronghold to a point."""

    distances = gm.distance(p, strongholds)
    return strongholds[distances.argmin(axis=-1, keepdims=True)].squeeze()

def points_in_cone(p: RectPoint, grid: RectCoordinates,
                   theta: ScalarLike, theta_err: ScalarLike = 0.05,
                   error_is_relative: bool = True) -> RectCoordinates:
    """
    Finds the possible stronghold locations from an Eye of Ender throw.
    
    Keyword arguments:
        `p`: the location of the player
        `grid`: the possible stronghold grid points to consider (see `generation_grid`)
        `theta`: the angle of the Eye of Ender throw
        `theta_err`: the error of the throw
        `error_is_relative`: whether `theta_err` is absolute or relative
    """

    r, phi = gm.to_polar(grid - p).T
    if error_is_relative:
        theta_b, theta_a = theta * (1 + gm.pm * theta_err)
    else:
        theta_b, theta_a = theta + gm.pm * theta_err

    targets = (r > 0) & gm.in_interval(phi, theta_a, theta_b)

    return grid[targets]
