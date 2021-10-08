from typing import Optional

import matplotlib
import numpy as np

from ..aliases import ArrayOfParameterVectors
from ..plot_utils import _check_and_create_fig_ax
from ..plots import (
    plot_line_through_points,
    plot_optimization_trajectory,
    plot_scatter_points,
)
from ..scans import plot_2D_scan_result
from ..scans.data_structures import Scan2DResult
from .data_structures import PCAobject


def plot_optimization_trajectory_on_pca(
    optimization_trajectory: ArrayOfParameterVectors,
    pca_object: PCAobject,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
):
    """Wrapper function around plot_optimization_trajectory to simplify
        plotting a on PCA scan.

    Args:
        optimization_trajectory: Parameter trajectory to be projected on a PCA scan.
        pca_object: PCAobject the contains a fitted PCA object, corresponding points
            and the components of interest.
        ax: Matplotlib axis to perform the plot on. If None, an axis is created
            from the current figure. Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.plot (plt.plot)
    """

    _, ax = _check_and_create_fig_ax(ax=ax)

    plot_optimization_trajectory(
        optimization_trajectory=optimization_trajectory,
        direction_x=pca_object.pca.components_[pca_object.components_ids[0]],
        direction_y=pca_object.pca.components_[pca_object.components_ids[1]],
        shift=pca_object.pca.mean_,
        ax=ax,
        **plot_kwargs,
    )


def plot_scatter_points_on_pca(
    scatter_points: ArrayOfParameterVectors,
    pca_object: PCAobject,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
):
    """Wrapper function around plot_scatter_points to simplify plotting a on PCA scan.

    Args:
        scatter_points: Parameter points to be projected on a PCA scan.
        pca_object: PCAobject the contains a fitted PCA object, corresponding points
            and the components of interest.
        ax: Matplotlib axis to perform the plot on. If None, an axis is created
            from the current figure. Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.scatter (plt.scatter)
    """

    _, ax = _check_and_create_fig_ax(ax=ax)

    plot_scatter_points(
        scatter_points=scatter_points,
        direction_x=pca_object.pca.components_[pca_object.components_ids[0]],
        direction_y=pca_object.pca.components_[pca_object.components_ids[1]],
        shift=pca_object.pca.mean_,
        ax=ax,
        **plot_kwargs,
    )


def plot_line_through_points_on_pca(
    points: ArrayOfParameterVectors,
    pca_object: PCAobject,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
):
    """Wrapper function around plot_line_through_points to simplify plotting a on PCA scan.

    Args:
        points: Parameter points to be projected on a PCA scan and plotted with a line.
        pca_object: PCAobject the contains a fitted PCA object, corresponding points
            and the components of interest.
        ax: Matplotlib axis to perform the plot on. If None, an axis is created
            from the current figure. Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.plot (plt.plot)
    """

    _, ax = _check_and_create_fig_ax(ax=ax)

    plot_line_through_points(
        points=points,
        direction_x=pca_object.pca.components_[pca_object.components_ids[0]],
        direction_y=pca_object.pca.components_[pca_object.components_ids[1]],
        shift=pca_object.pca.mean_,
        ax=ax,
        **plot_kwargs,
    )


def plot_pca_landscape(
    scan_result: Scan2DResult,
    pca_object: PCAobject,
    fig: Optional[matplotlib.figure.Figure] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
):
    """Wrapper function around plot_2D_scan_result to simplify plotting a on PCA scan.

    Args:
        scan_result: Scan2DResult object from a performed 2D scan.
        pca: sklearn PCA object fitted on the optimization trajectory.
        fig: Matplotlib figure to perform the plot on. If None, a new figure
            and axis are created. Defaults to None.
        ax: Matplotlib axis to perform the plot on. If None, a new axis is created
            from the current figure. Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.pcolormesh
            (plt.pcolormesh)
    """
    fig, ax = _check_and_create_fig_ax(fig=fig, ax=ax)

    plot_2D_scan_result(
        scan_result,
        fig=fig,
        ax=ax,
        **plot_kwargs,
    )

    component1 = pca_object.components_ids[0]
    component2 = pca_object.components_ids[1]
    ax.set_xlabel(
        "{}. component explains {:.2f}% of variance".format(
            component1, pca_object.pca.explained_variance_ratio_[component1] * 100
        )
    )
    ax.set_ylabel(
        "{}. component explains {:.2f}% of variance".format(
            component2, pca_object.pca.explained_variance_ratio_[component2] * 100
        )
    )
