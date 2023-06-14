import math
import warnings
from typing import NamedTuple, Optional, Tuple

import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator

from orqviz.scans.data_structures import Scan2DResult

from .aliases import DirectionVector, LossFunction, ParameterVector
from .plot_utils import _check_and_create_fig_ax
from .scans.scans_2D import perform_2D_scan


class FourierResult(NamedTuple):
    """Datatype for 2D Fourier scans to combine the scan result and scan instruction."""

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

    Args:
        origin: Origin point of the 2D scan.
        loss_function: Function to perform the scan on. It must receive only a
            numpy.ndarray of parameters, and return a real number.
            If your function requires more arguments, consider using the
            'LossFunctionWrapper' class from 'orqviz.loss_function'.
        direction_x: x-direction vector for scan. Has same shape as origin.
            If None, a random unit vector is sampled. Defaults to None.
        direction_y: y-direction vector for scan. Has same shape as origin.
            If None, a random unit vector is sampled. Defaults to None.
        n_steps_x: Number of points evaluated along the x-direction. Defaults to 20.
        n_steps_y: Number of points evaluated along the y-direction.
            If None, set value to n_steps_x. Defaults to None.
        end_points_x: Range of scan along the x-direction in units of direction_x.
            Defaults to (0, 2pi).
        end_points_y: Range of scan along the y-direction in units of direction_y.
            Defaults to (0, 2pi).

    Returns:
        FourierResult with the following attributes:
            values: Output from np.fft.rfft2 with the format of numpy's output, which is
                the coefficient for 0, then the coefficients of positive frequencies in
                increasing order by frequency, then the coefficients of negative
                frequences in increasing order.
            end_points_x: End points of the scan along the x-direction.
            end_points_y: End points of the scan along the y-direction.
    """
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
    return perform_2D_fourier_transform(scan2D_result, end_points_x, end_points_y)


def perform_2D_fourier_transform(
    scan2D_result: Scan2DResult,
    end_points_x: Tuple[float, float],
    end_points_y: Tuple[float, float],
):
    """Performs a discrete real fourier transform on an already completed scan of a
    loss function.

    Args:
        scan2D_result: Result of a 2D scan. Output of orqviz.scans.perform_2D_scan.
        end_points_x: Range used for the scan along the x-direction.
        end_points_y: Range used for the scan along they-direction.
    """
    fourier_result = np.fft.fftshift(np.fft.fftn(scan2D_result.values, norm="forward"))
    return FourierResult(fourier_result, end_points_x, end_points_y)


def plot_2D_fourier_result(
    fourier_result: FourierResult,
    max_freq_x: int,
    max_freq_y: int,
    show_full_spectrum: bool = False,
    remove_constant_term: bool = True,
    fig: Optional[matplotlib.figure.Figure] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **plot_kwargs,
):
    """Plots a 2D fourier result.
    Note: Fourier coefficients are complex numbers. However, due to the inability to
    plot complex numbers on real axes, the magnitudes of the coefficients are plotted.
    This means that phase has no influence on the visual output of the plot.

    Note: If the plot appears distorted, most often the reason for that is the choice
    of max_freq_x or max_freq_y.

    Args:
        fourier_result: Fourier result to be plotted.
        max_freq_x: Maximum frequency to be plotted along the x-direction.
        max_freq_y: Maximum frequency to be plotted along the y-direction.
        show_full_spectrum: If true plots the whole spectrum (including the
            redundant part). Defaults to False.
        remove_constant_term: If true sets the value of (0, 0) frequency (the constant
            term) to 0. Defaults to True.
        fig: Matplotlib figure to perfom a plot on. If None, a new figure
            and axis are created from the current figure. Defaults to None.
        ax: Matplotlib axis to perform plot on. If None, a new axis is created.
            Defaults to None.
        plot_kwargs: kwargs for plotting with matplotlib.pyplot.pcolormesh
            (plt.pcolormesh)
    """
    raw_values = fourier_result.values
    size = raw_values.shape[0]

    half = int(size / 2)
    if show_full_spectrum:
        min_x = int(half - max_freq_x)
    else:
        min_x = half
    max_x = int(half + max_freq_x)
    min_y = int(half - max_freq_y) - 1
    max_y = int(half + max_freq_y) + 1
    result = fourier_result
    plottable_result = np.abs(result.values)

    if remove_constant_term:
        plottable_result[half][half] = 0
    truncated_result = plottable_result[min_y:max_y, min_x:max_x]

    n_x = truncated_result.shape[1]
    n_y = truncated_result.shape[0]

    if show_full_spectrum:
        x_axis = np.arange(-n_x / 2, n_x / 2)
    else:
        x_axis = np.arange(0, n_x)
    y_axis = np.arange(-n_y / 2, n_y / 2)

    XX, YY = np.meshgrid(x_axis, y_axis)

    default_plot_kwargs = {"shading": "auto"}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}
    fig, ax = _check_and_create_fig_ax(fig=fig, ax=ax)
    mesh_plot = ax.pcolormesh(XX, YY, truncated_result, **plot_kwargs, rasterized=True)

    fig.colorbar(mesh_plot, ax=ax)
    ax.set_xlabel("Frequencies x")
    ax.set_ylabel("Frequencies y")
    ax.set_ylim(-max_freq_y, max_freq_y)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def inverse_fourier(result: FourierResult) -> FourierResult:
    """Inverts a Fourier result.
    Args:
        result: Fourier result to be inverted.
    Returns:
        Inverted result.
    """
    return FourierResult(
        np.fft.ifftn(np.fft.ifftshift(result.values), norm="forward"),
        result.end_points_x,
        result.end_points_y,
    )


def plot_inverse_fourier_result(
    result: FourierResult,
    fig=None,
    ax=None,
    **plot_kwargs,
):
    """Plots inverse fourier result according to end points.
    Args:
        result: output of `orqviz.fourier.inverse_fourier`
    """
    Nx = result.values.shape[1]
    Ny = result.values.shape[0]
    x_axis = (
        np.arange(Nx) / Nx * (result.end_points_x[1] - result.end_points_x[0])
        + result.end_points_x[0]
    )
    y_axis = (
        np.arange(Ny) / Ny * (result.end_points_y[1] - result.end_points_y[0])
        + result.end_points_y[0]
    )
    XX, YY = np.meshgrid(x_axis, y_axis)

    default_plot_kwargs = {"shading": "auto"}
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    fig, ax = _check_and_create_fig_ax(fig=fig, ax=ax)
    # if it was rfft then only the real components matter.
    plottable_result = np.real(result.values)
    mesh_plot = ax.pcolormesh(XX, YY, plottable_result, **plot_kwargs, rasterized=True)
    fig.colorbar(mesh_plot, ax=ax)
    ax.set_xlabel("Scan Direction x")
    ax.set_ylabel("Scan Direction y")
