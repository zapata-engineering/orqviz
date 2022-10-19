import matplotlib.pyplot as plt
import numpy as np

from orqviz.scans import perform_1D_scan, perform_2D_interpolation
from orqviz.scans.plots import plot_1D_scan_result, plot_2D_interpolation_result


def SUM_OF_SINES(params):
    return np.sum(np.sin(params))


def test_plot_1D_scan_result():
    origin = np.array([0.0, 0.0])
    direction_x = np.array([1.0, 0.0])
    end_points_x = (-np.pi, np.pi)
    n_steps_x = 100

    scan_1d = perform_1D_scan(
        origin=origin,
        loss_function=SUM_OF_SINES,
        direction=direction_x,
        n_steps=n_steps_x,
        end_points=end_points_x,
    )

    plot_1D_scan_result(scan_1d)
    fig, ax = plt.subplots()
    plot_1D_scan_result(scan_1d, ax)


def test_plot_2D_interpolation_result():
    point_1 = np.array([1, 0])
    point_2 = np.array([0, 2])
    end_points_x = (-0.5, 1.5)
    end_points_y = (-0.5, 0.5)
    n_steps_x = 100
    n_steps_y = 10

    scan_2d = perform_2D_interpolation(
        point_1=point_1,
        point_2=point_2,
        loss_function=SUM_OF_SINES,
        n_steps_x=n_steps_x,
        n_steps_y=n_steps_y,
        end_points_x=end_points_x,
        end_points_y=end_points_y,
    )

    plot_2D_interpolation_result(scan_2d)
    fig, ax = plt.subplots()
    plot_2D_interpolation_result(scan_2d, fig, ax)
