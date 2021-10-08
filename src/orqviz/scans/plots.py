from typing import Optional

import matplotlib
import numpy as np

# this import is unused but solves issues with older matplotlib versions and 3d plots
from mpl_toolkits.mplot3d import Axes3D

from ..plot_utils import _check_and_create_3D_ax, _check_and_create_fig_ax
from .data_structures import Scan1DResult, Scan2DResult


def plot_1D_scan_result(
    scan1d_result: Scan1DResult,
    ax: Optional[matplotlib.axes.Axes] = None,
    in_units_of_direction: bool = False,
    **plot_kwargs,
) -> None:
    """Function to plot a Scan1DResult.

    Args:
        scan1d_result: Scan result to be plotted.
        ax: Matplotlib axis to perform plot on.
            If None, a new axis is created from the current figure. Defaults to None.
        in_units_of_direction: Flag to indicate axis ticks are in units
            of the direction. If False, ticks are in units of Euclidean distance.
            Defaults to False.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.plot (plt.plot)
    """
    _, ax = _check_and_create_fig_ax(ax=ax)

    default_plot_kwargs = {"linewidth": 3}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    ax.plot(
        scan1d_result._get_coordinates_on_direction(
            in_units_of_direction=in_units_of_direction
        ),
        scan1d_result.values,
        **plot_kwargs,
    )
    ax.set_xlabel("Scan Direction")
    ax.set_ylabel("Loss Value")


def plot_1D_interpolation_result(
    interpolation_result: Scan1DResult,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
) -> None:
    """Function to plot a Scan1DResult of a 1D interpolation.

    Args:
        interpolation_result: Interpolation scan result to be plotted.
        ax: Matplotlib axis to perform plot on. If None, a new axis is created
        from the current figure. Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.plot (plt.plot)
    """
    _, ax = _check_and_create_fig_ax(ax=ax)

    default_plot_kwargs = {"color": "royalblue"}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    plot_1D_scan_result(
        scan1d_result=interpolation_result,
        ax=ax,
        in_units_of_direction=True,
        **plot_kwargs,
    )
    ax.axvline(x=0.0, color="red")
    ax.axvline(x=1.0, color="red")
    ax.set_xlabel("Interpolation Direction")
    ax.set_ylabel("Loss Value")


def plot_2D_scan_result(
    scan2d_result: Scan2DResult,
    fig: Optional[matplotlib.figure.Figure] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    in_units_of_direction: bool = False,
    **plot_kwargs,
) -> None:
    """Function to plot a Scan2DResult.

    Args:
        scan2d_result: Scan result to be plotted.
        fig: Matplotlib figure to perfom a plot on.
            If None, a new figure and axis are created. Defaults to None.
        ax: Matplotlib axis to perform plot on. If None, a new axis is created
            from the current figure. Defaults to None.
        in_units_of_direction: Flag to indicate axis ticks are in units
            of the direction. If False, ticks are in units of Euclidean distance.
            Defaults to False.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.pcolormesh
            (plt.pcolormesh)
    """
    fig, ax = _check_and_create_fig_ax(fig=fig, ax=ax)
    x, y = scan2d_result._get_coordinates_on_directions(
        in_units_of_direction=in_units_of_direction
    )
    XX, YY = np.meshgrid(x, y)

    default_plot_kwargs = {"shading": "auto"}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    mesh_plot = ax.pcolormesh(
        XX, YY, scan2d_result.values, **plot_kwargs, rasterized=True
    )
    fig.colorbar(mesh_plot, ax=ax)
    ax.set_xlabel("Scan Direction x")
    ax.set_ylabel("Scan Direction y")


def plot_2D_interpolation_result(
    scan2D_result: Scan2DResult,
    fig: Optional[matplotlib.figure.Figure] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
) -> None:
    """Function to plot a Scan2DResult from 2D interpolation.

    Args:
        scan2d_result: Scan result to be plotted.
        fig: Matplotlib figure to perfom a plot on. If None, a new figure
            and axis are created from the current figure. Defaults to None.
        ax: Matplotlib axis to perform plot on. If None, a new axis is created.
            Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.pcolormesh
            (plt.pcolormesh)
    """
    fig, ax = _check_and_create_fig_ax(fig=fig, ax=ax)

    plot_2D_scan_result(
        scan2d_result=scan2D_result,
        fig=fig,
        ax=ax,
        in_units_of_direction=True,
        **plot_kwargs,
    )
    ax.scatter(0, 0, color="red", linewidth=2)
    ax.scatter(1, 0, color="red", linewidth=2)
    ax.set_xlabel("Interpolation Direction")
    ax.set_ylabel("Scan Direction y")


def plot_2D_scan_result_as_3D(
    scan2D_result: Scan2DResult,
    ax: Optional[matplotlib.axes.Axes] = None,
    in_units_of_direction: bool = False,
    **plot_kwargs,
) -> None:
    """Function to create a 3D plot from a Scan2DResult.

    Args:
        scan2d_result: Scan result to be plotted.
        ax: Matplotlib axis to perform plot on. Must have projection='3d'
            set on creation. If None, a new axis is created. Defaults to None.
        in_units_of_direction: Flag to indicate axis ticks are in units
            of the direction. If False, ticks are in units of Euclidean distance.
            Defaults to False.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.plot_surface
            (plt.plot_surface)
    """
    ax = _check_and_create_3D_ax(ax=ax)

    x, y = scan2D_result._get_coordinates_on_directions(
        in_units_of_direction=in_units_of_direction
    )
    XX, YY = np.meshgrid(x, y)

    plot_kwargs_defaults = {"cmap": "viridis", "alpha": 0.8}
    plot_kwargs = {**plot_kwargs_defaults, **plot_kwargs}

    ax.plot_surface(XX, YY, scan2D_result.values, **plot_kwargs)

    ax.view_init(elev=35, azim=-70)

    ax.set_xlabel("Scan Direction x")
    ax.set_ylabel("Scan Direction y")
    ax.set_zlabel("Loss Value")
