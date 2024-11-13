from . import math as gm, types

__all__ = ["closest_stronghold", "points_in_cone"]

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
    N = (None for _ in range(s.ndim))
    P = p[..., *N]

    # finds the mins along the last axis:
    # if p is an array, then it 
    # i.e. for each `Coordinates` in a `CoordinateSets`,
    # and for 
    mins = gm.distance(s, P).min(axis=-1, keepdims=True)

    S = gm.np.tile(s, (*p.shape, *gm.np.ones_like(s.shape)))

    D = gm.np.where(gm.distance(S, P) == mins, S, 0)
    return D.sum(axis=-1)

def points_in_cone(player: types.Point, grid: types.Coordinates,
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

    grid_rel = grid - player
    r, phi = gm.radius(grid_rel), gm.angle(grid_rel)
    if error_is_relative:
        theta_b, theta_a = theta * (1 + gm.pm * theta_err)
    else:
        theta_b, theta_a = theta + gm.pm * theta_err

    target_mask = (r > 0) & gm.in_interval(phi, theta_a, theta_b)

    return grid[target_mask]
