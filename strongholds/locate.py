from . import math as gm, types

__all__ = ["closest_stronghold", "points_in_cone"]

def closest_stronghold(p: types.Point, strongholds: types.Coordinates) -> types.Point:
    """Finds the closest stronghold to a point."""

    distances = gm.distance(p, strongholds)
    args = distances.argmin(axis=-1, keepdims=True)
    closest = gm.np.take_along_axis(strongholds, args, axis=-1).squeeze()

    if strongholds.ndim == 1:
        return closest.item()
    return closest

def points_in_cone(p: types.Point, grid: types.Coordinates,
                   theta: types.ScalarLike, theta_err: types.ScalarLike = 0.05,
                   error_is_relative: bool = False) -> types.Coordinates:
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
