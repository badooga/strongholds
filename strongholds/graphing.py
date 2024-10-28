import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np

@np.vectorize
def setup_xz_plot(ax: Axes) -> Axes:
    """Labels axes."""

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    return ax

def xz_subplots(*args, **kwargs) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(*args, **kwargs, constrained_layout=True)
    ax = setup_xz_plot(ax)
    if not ax.shape:
        ax = ax.item()
    return fig, ax

def flip_zaxis(*args: Axes) -> None:
    """Inverts the vertical axis (as north is -z)."""
    for ax in args:
        ax.set_ylim(ax.get_ylim()[::-1])
