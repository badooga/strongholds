from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

__all__ = ["setup_xz_plot", "xz_subplots", "flip_zaxis"]

ring_colors = ["red", "orange", "yellow", "green",
               "blue", "indigo", "darkviolet", "violet"]


@np.vectorize
def setup_xz_plot(ax: Axes) -> Axes:
    """Labels axes."""

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    return ax


def xz_subplots(*args, **fkwargs) -> tuple[Figure, Axes]:
    """Sets up an x-z plot."""

    kwargs = {"constrained_layout": True}
    kwargs.update(fkwargs)

    fig, ax = plt.subplots(*args, **kwargs)
    ax = setup_xz_plot(ax).tolist()
    return fig, ax


def flip_zaxis(*args: Axes) -> None:
    """Inverts the vertical axis (as north is -z)."""
    for ax in args:
        ax.set_ylim(ax.get_ylim()[::-1])
