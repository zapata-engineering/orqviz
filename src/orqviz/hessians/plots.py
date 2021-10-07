from typing import Callable, List, Optional, Tuple

import matplotlib
import numpy as np

from ..plot_utils import _check_and_create_fig_ax
from ..scans import Scan1DResult, perform_1D_scan
from ..scans.plots import plot_1D_scan_result


def plot_1D_hessian_eigenvector_scan_result(
    list_of_scans: List[Scan1DResult],
    eigenvalues: Optional[np.ndarray] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
):
    """Function to plot 1D scans along all eigenvecor directions of a Hessian

    Args:
        list_of_scans: List of Scan1DResult that
        ax: Matplotlib axis to perform the plot on.
            If None, a new axis is created from the current figure. Defaults to None.
        n_points: Number of points to evaluate the loss along each direction.
            Defaults to 31.
        endpoints: End points for scan along each direction.
            Defaults to (-np.pi, np.pi).
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.plot (plt.plot)
    """
    _, ax = _check_and_create_fig_ax(ax=ax)

    for ii in range(len(list_of_scans)):
        plot_1D_scan_result(
            list_of_scans[ii],
            ax,
            label="{:.3f}".format(eigenvalues[ii]) if eigenvalues is not None else "",
            **plot_kwargs,
        )

    ax.set_xlabel("Eigenvector direction")
    ax.set_ylabel("Loss Value")
    if eigenvalues is not None:
        ax.legend()
