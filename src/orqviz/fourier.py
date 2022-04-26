from .scans.scans_2D import perform_2D_scan
import numpy as np
from typing import Optional, Tuple

import numpy as np

from .aliases import (
    DirectionVector,
    LossFunction,
    ParameterVector,
)
import warnings
import matplotlib
from typing import NamedTuple
from .plot_utils import _check_and_create_fig_ax


class FourierResult(NamedTuple):
    """Datatype for 2D scans to combine the scan result and scan instruction.

    values: Output from np.fft.rfft2 with the specific format numpy has which is the coefficient for 0, then postive coefficients, then negative coefficients in decreasing order.
    """

    values: np.ndarray
    end_points_x: Tuple[float, float]
    end_points_y: Tuple[float, float]


def scan_2D_fourier(
    origin: ParameterVector,
    loss_function: LossFunction,
    direction_x: Optional[DirectionVector] = None,
    direction_y: Optional[DirectionVector] = None,
    n_steps_x: int = 20,
    n_steps_y: Optional[int] = None,
    end_points_x: Tuple[float, float] = (0, 2 * np.pi),
    end_points_y: Optional[Tuple[float, float]] = None,
) -> FourierResult:
    """Performs a discrete real fourier transform on the 2D scan of a loss function.
    Constrained to loss functions with real output.
    """
    if n_steps_y is None:
        n_steps_y = n_steps_x
    if end_points_y is None:
        end_points_y = end_points_x
    scan2D_result = perform_2D_scan(
        origin,
        loss_function,
        direction_x=direction_x,
        direction_y=direction_y,
        n_steps_x=n_steps_x,
        n_steps_y=n_steps_y,
        end_points_x=end_points_x,
        end_points_y=end_points_y,
    )
    fourier_result = np.fft.rfft2(scan2D_result.values, norm="forward")
    return FourierResult(fourier_result, end_points_x, end_points_y)


def plot_2D_fourier_result(
    result: FourierResult,
    max_freq_x: int = None,
    max_freq_y: int = None,
    show_negative_frequencies: bool = False,
    fig: Optional[matplotlib.figure.Figure] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
):
    """Plots a 2D fourier result.
    Args:
        result: Fourier result to be plotted.
        show_negative_frequencies: only plot positive frequencies if False
        fig: Matplotlib figure to perfom a plot on. If None, a new figure
            and axis are created from the current figure. Defaults to None.
        ax: Matplotlib axis to perform plot on. If None, a new axis is created.
            Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.pcolormesh
            (plt.pcolormesh)

    Note: due to inability to plot complex numbers, the magnitude of complex coefficients are plotted. This means that phase has no influence on plotted results.
    """
    # Do a swappy thingy st frequencies are lined up in order from least to greatest
    plottable_result = np.abs(result.values)
    Nx = result.values.shape[1]
    Ny = result.values.shape[0]
    if max_freq_y is None:
        max_freq_y = min(Ny // 2, max_freq_x or np.inf)
    if max_freq_x is None:
        max_freq_x = Nx - 1

    # normalize frequencies for range bigger than 1 period of 2pi
    # (note that this is different from normalizing the coefficient magnitudes)
    norm_x = (result.end_points_x[1] - result.end_points_x[0]) / (2 * np.pi)
    norm_y = (result.end_points_y[1] - result.end_points_y[0]) / (2 * np.pi)
    if max_freq_x > (Nx - 1) / norm_x:
        warnings.warn(
            "Max x frequency is too high for the number of steps so the default will be used."
        )
        max_freq_x = (Nx - 1) / norm_x

    if max_freq_y > Ny // 2 / norm_y:
        warnings.warn(
            "Max y frequency is too high for the number of steps so the default will be used."
        )
        max_freq_y = Ny // 2 / norm_y

    if show_negative_frequencies:
        truncated_result = _truncate_result_according_to_resolution(
            _swap(plottable_result), int(max_freq_x * norm_x), int(max_freq_y * norm_y)
        )
        y_axis = (
            np.arange(
                (
                    -max_freq_y
                    if truncated_result.shape[0] % 2 == 1
                    else -max_freq_y + 1
                ),
                max_freq_y * int(norm_y) + 1,
            )
            / norm_y
        )  # this shit only works for odd truncated_result.shape[0]
    else:
        truncated_result = plottable_result[
            : int(max_freq_y * norm_y) + 1, : int(max_freq_x * norm_x) + 1
        ]
        y_axis = np.arange(0, max_freq_y * int(norm_y) + 1) / norm_y

    x_axis = np.arange(0, max_freq_x * int(norm_x) + 1) / norm_x
    # you want the extra for the positive side
    XX, YY = np.meshgrid(x_axis, y_axis)

    default_plot_kwargs = {"shading": "auto"}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    fig, ax = _check_and_create_fig_ax(fig=fig, ax=ax)
    mesh_plot = ax.pcolormesh(XX, YY, truncated_result, **plot_kwargs, rasterized=True)
    fig.colorbar(mesh_plot, ax=ax)
    ax.set_xlabel("Scan Direction x")
    ax.set_ylabel("Scan Direction y")


def _truncate_result_according_to_resolution(
    result: np.ndarray, res_x: int, res_y: int
):
    """resolution args are uh not actual resolution but resolution on 1 side of the center. the returned reesult is of size 2 * res + 1 in y dir and res+1 on x dir."""
    cy = (result.shape[0] - 1) // 2  # center
    return result[cy - res_y : cy + res_y + 1, 0 : res_x + 1]
    # Note this always makes result.shape[0] odd


def _swap(result: np.ndarray) -> np.ndarray:
    Ny = result.shape[0]
    return np.append(result[Ny // 2 + 1 :], result[0 : Ny // 2 + 1], axis=0)


def _iswap(result: np.ndarray) -> np.ndarray:
    Ny = result.shape[0]
    return np.append(result[(Ny - 1) // 2 :], result[0 : (Ny - 1) // 2], axis=0)


def plot_inverse_fourier_result(
    result: np.ndarray,
    end_points_x,
    end_points_y=None,
    fig=None,
    ax=None,
    **plot_kwargs,
):
    """Plots inverse fourier result according to end points.
    Args:
        result: output of np.fft.irfft2
    """
    if end_points_y is None:
        end_points_y = end_points_x
    Nx = result.shape[1]
    Ny = result.shape[0]
    x_axis = np.arange(Nx) / Nx * (end_points_x[1] - end_points_x[0]) + end_points_x[0]
    y_axis = np.arange(Ny) / Ny * (end_points_y[1] - end_points_y[0]) + end_points_y[0]
    XX, YY = np.meshgrid(x_axis, y_axis)

    default_plot_kwargs = {"shading": "auto"}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    fig, ax = _check_and_create_fig_ax(fig=fig, ax=ax)
    # if it was rfft then only the real components matter.
    plottable_result = np.real(result)
    mesh_plot = ax.pcolormesh(XX, YY, plottable_result, **plot_kwargs, rasterized=True)
    fig.colorbar(mesh_plot, ax=ax)
    ax.set_xlabel("Scan Direction x")
    ax.set_ylabel("Scan Direction y")