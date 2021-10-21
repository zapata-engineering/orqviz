import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from orqviz.scans import (
    get_2D_slice_around_point,
    perform_1D_interpolation,
    perform_1D_scan,
    perform_2D_interpolation,
)
from orqviz.scans.data_structures import Scan1DResult, Scan2DResult
from orqviz.scans.plots import (
    plot_1D_scan_result,
    plot_2D_interpolation_result,
    plot_2D_scan_result,
)
from orqviz.utils import load_viz_object, save_viz_object


def SUM_OF_SINES(params):
    return np.sum(np.sin(params))


def test_1D_scan():
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
    save_viz_object(scan_1d, "test")
    loaded_scan1d = load_viz_object("test")
    os.remove("test")

    for scan_results in [scan_1d, loaded_scan1d]:
        assert isinstance(scan_results, Scan1DResult)
        assert len(scan_results.values) == n_steps_x
        np.testing.assert_equal(scan_results.direction, direction_x)
        assert scan_results.params_list.shape == (n_steps_x, 2)

        np.testing.assert_equal(
            scan_results.params_list[0],
            origin + direction_x * end_points_x[0],
        )
        np.testing.assert_equal(
            scan_results.params_list[-1],
            origin + direction_x * end_points_x[1],
        )


def test_1D_interpolation():
    point1 = np.array([1, 0])
    point2 = np.array([0, 1])
    direction_x = point2 - point1
    end_points_x = (0, 1)
    n_steps_x = 23

    scan_1d = perform_1D_interpolation(
        point_1=point1,
        point_2=point2,
        loss_function=SUM_OF_SINES,
        n_steps=n_steps_x,
        end_points=end_points_x,
    )
    save_viz_object(scan_1d, "test")
    loaded_scan1d = load_viz_object("test")
    os.remove("test")

    for scan_results in [scan_1d, loaded_scan1d]:
        assert isinstance(scan_results, Scan1DResult)
        assert len(scan_results.values) == n_steps_x
        np.testing.assert_equal(scan_results.direction, direction_x)
        assert scan_results.params_list.shape == (n_steps_x, 2)

        np.testing.assert_equal(
            scan_results.params_list[0],
            point1,
        )
        np.testing.assert_equal(
            scan_results.params_list[-1],
            point2,
        )


def test_get_2D_slice_around_point_in_2D_space():
    origin = np.array([0, 0])
    direction_x = np.array([1, 0])
    direction_y = np.array([0, 1])
    end_points_x = (-np.pi, np.pi)
    end_points_y = (-np.pi / 2, np.pi / 2)
    n_steps_x = 100
    n_steps_y = 10

    scan_2d = get_2D_slice_around_point(
        origin=origin,
        loss_function=SUM_OF_SINES,
        direction_x=direction_x,
        direction_y=direction_y,
        n_steps_x=n_steps_x,
        n_steps_y=n_steps_y,
        end_points_x=end_points_x,
        end_points_y=end_points_y,
    )
    save_viz_object(scan_2d, "test")
    loaded_scan2d = load_viz_object("test")
    os.remove("test")

    for scan_results in [scan_2d, loaded_scan2d]:
        assert isinstance(scan_results, Scan2DResult)
        assert scan_results.values.shape == (n_steps_y, n_steps_x)
        np.testing.assert_equal(scan_results.direction_x, direction_x)
        np.testing.assert_equal(scan_results.direction_y, direction_y)
        assert scan_results.params_grid.shape == (n_steps_y, n_steps_x, 2)

        for i, j in [(0, 0), (0, 1), (1, 0), (0, 1)]:
            np.testing.assert_equal(
                scan_results.params_grid[-i][-j],
                origin + end_points_x[j] * direction_x + end_points_y[i] * direction_y,
            )


def test_get_2D_slice_around_point_in_5D_space():
    origin = np.zeros(5)
    direction_x = np.array([0, 0, 0, 1, 0])
    direction_y = np.array([0, 0, 0, 0, 1])
    end_points_x = (-np.pi, np.pi)
    end_points_y = (-np.pi / 2, np.pi / 2)
    n_steps_x = 100
    n_steps_y = 20

    scan_2d = get_2D_slice_around_point(
        origin=origin,
        loss_function=SUM_OF_SINES,
        direction_x=direction_x,
        direction_y=direction_y,
        n_steps_x=n_steps_x,
        n_steps_y=n_steps_y,
        end_points_x=end_points_x,
        end_points_y=end_points_y,
    )

    save_viz_object(scan_2d, "test")
    loaded_scan2d = load_viz_object("test")
    os.remove("test")

    for scan_results in [scan_2d, loaded_scan2d]:
        assert isinstance(scan_results, Scan2DResult)
        assert scan_results.values.shape == (n_steps_y, n_steps_x)
        np.testing.assert_equal(scan_results.direction_x, direction_x)
        np.testing.assert_equal(scan_results.direction_y, direction_y)
        assert scan_results.params_grid.shape == (n_steps_y, n_steps_x, 5)

        for i, j in [(0, 0), (0, 1), (1, 0), (0, 1)]:
            np.testing.assert_equal(
                scan_results.params_grid[-i][-j],
                origin + end_points_x[j] * direction_x + end_points_y[i] * direction_y,
            )


def test_interpolation_2D_in_2D_space():
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

    save_viz_object(scan_2d, "test")
    loaded_scan2d = load_viz_object("test")
    os.remove("test")

    for scan_results in [scan_2d, loaded_scan2d]:
        assert isinstance(scan_results, Scan2DResult)
        assert scan_results.values.shape == (n_steps_y, n_steps_x)
        np.testing.assert_equal(scan_results.direction_x, point_2 - point_1)
        assert np.dot(
            scan_results.direction_x, scan_results.direction_y
        ) == pytest.approx(0)
        assert scan_results.params_grid.shape == (n_steps_y, n_steps_x, 2)
