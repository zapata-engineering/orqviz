from typing import Optional

import matplotlib
import numpy as np

from .aliases import ArrayOfParameterVectors
from .geometric import get_coordinates_on_direction
from .plot_utils import _check_and_create_fig_ax


def plot_optimization_trajectory(
    optimization_trajectory: ArrayOfParameterVectors,
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    ax: matplotlib.axes.Axes,
    shift: Optional[np.ndarray] = None,
    **plot_kwargs,
):
    """Function to project and plot a parameter trajectory on a 2D plane.

    Args:
        optimization_trajectory: Parameter trajectory to be projected
            and plotted on a 2D plane.
        direction_x: x-direction of the 2D plane.
        direction_y: y-direction of the 2D plane.
        ax: Matplotlib axis to perform plot on. If None, a new axis
            is created from the current figure. Defaults to None.
        shift: Origin to shift the trajectory to. Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.plot (plt.plot)
    """

    _, ax = _check_and_create_fig_ax(ax=ax)

    projected_trajectory_x = get_coordinates_on_direction(
        optimization_trajectory, direction_x, origin=shift
    )
    projected_trajectory_y = get_coordinates_on_direction(
        optimization_trajectory, direction_y, origin=shift
    )

    default_plot_kwargs = {
        "color": "lightgray",
        "linestyle": "-",
        "marker": ".",
        "alpha": 0.8,
    }
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    ax.plot(
        projected_trajectory_x,
        projected_trajectory_y,
        **plot_kwargs,
    )
    ax.plot(
        projected_trajectory_x[0],
        projected_trajectory_y[0],
        c=plot_kwargs["color"],
        marker="s",
        alpha=plot_kwargs["alpha"],
    )
    ax.plot(
        projected_trajectory_x[-1],
        projected_trajectory_y[-1],
        c=plot_kwargs["color"],
        marker="*",
        alpha=plot_kwargs["alpha"],
    )


def plot_scatter_points(
    scatter_points: ArrayOfParameterVectors,
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    ax: Optional[matplotlib.axes.Axes] = None,
    shift: Optional[np.ndarray] = None,
    **plot_kwargs,
):
    """Function to project and scatter plot an array of points on a 2D plane.

    Args:
        scatter_points: Points to be to be projected and scattered on a 2D plane.
        direction_x: x-direction of the 2D plane.
        direction_y: y-direction of the 2D plane.
        ax: Matplotlib axis to perform plot on. If None, a new axis
            is created from the current figure. Defaults to None.
        shift: Origin to shift the trajectory to. Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.scatter (plt.scatter)
    """

    _, ax = _check_and_create_fig_ax(ax=ax)

    projected_scatter_x = get_coordinates_on_direction(
        scatter_points, direction_x, origin=shift
    )
    projected_scatter_y = get_coordinates_on_direction(
        scatter_points, direction_y, origin=shift
    )

    default_plot_kwargs = {"color": "lightgray", "marker": ".", "alpha": 0.8}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    ax.scatter(
        projected_scatter_x,
        projected_scatter_y,
        **plot_kwargs,
    )


def plot_line_through_points(
    points: ArrayOfParameterVectors,
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    ax: Optional[matplotlib.axes.Axes] = None,
    shift: Optional[np.ndarray] = None,
    **plot_kwargs,
):
    """Function to project and points on a 2D plane and plot a line through them.

    Args:
        points: Points to be to be projected on a 2D plane and plot a line through.
        direction_x: x-direction of the 2D plane.
        direction_y: y-direction of the 2D plane.
        ax: Matplotlib axis to perform plot on. If None, a new axis is created
            from the current figure. Defaults to None.
        shift: Origin to shift the trajectory to. Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.plot (plt.plot)
    """

    _, ax = _check_and_create_fig_ax(ax=ax)

    projected_scatter_x = get_coordinates_on_direction(
        points, direction_x, origin=shift
    )
    projected_scatter_y = get_coordinates_on_direction(
        points, direction_y, origin=shift
    )

    default_plot_kwargs = {"color": "lightgray", "alpha": 0.8}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    ax.plot(
        projected_scatter_x,
        projected_scatter_y,
        **plot_kwargs,
    )
