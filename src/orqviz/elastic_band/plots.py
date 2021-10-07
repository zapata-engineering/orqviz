from typing import Callable, List, Optional

import matplotlib
import numpy as np

from ..aliases import ParameterVector
from ..plot_utils import _check_and_create_fig_ax
from ..scans import eval_points_on_path
from .neb import Chain


def plot_all_chains_losses(
    all_chains: List[Chain],
    loss_function: Callable[[ParameterVector], float],
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
) -> None:
    """Function to plot

    Args:
        all_chains: List of Chains to evaluate the loss on.
        loss_function: Loss function to evaluate the Chains
        ax: Matplotlib axis to plot on. If None, a new axis is created
            from the current figure. Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.plot (plt.plot)
    """
    _, ax = _check_and_create_fig_ax(ax=ax)

    for chain in all_chains:
        losses = eval_points_on_path(chain.pivots, loss_function)
        inter_points = chain.get_weights()
        ax.plot(inter_points, losses, **plot_kwargs)

    ax.set_xlabel("Chain Position")
    ax.set_ylabel("Loss Value")
