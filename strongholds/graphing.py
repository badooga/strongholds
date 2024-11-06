from matplotlib import pyplot as plt, Axes, Figure
import numpy as np

__all__ = ["setup_xz_plot", "xz_subplots", "flip_zaxis"]

ring_colors = ["red", "orange", "yellow", "green", "blue", "indigo", "darkviolet", "violet"]

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
    ax = setup_xz_plot(ax)
    if not ax.shape:
        ax = ax.item()
    return fig, ax

def flip_zaxis(*args: Axes) -> None:
    """Inverts the vertical axis (as north is -z)."""
    for ax in args:
        ax.set_ylim(ax.get_ylim()[::-1])
