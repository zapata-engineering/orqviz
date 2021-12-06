import numpy as np
import pytest

from orqviz.geometric import (
    direction_linspace,
    get_coordinates_on_direction,
    get_random_normal_vector,
    get_random_orthonormal_vector,
    relative_periodic_trajectory_wrap,
    relative_periodic_wrap,
    uniformly_distribute_trajectory,
)


@pytest.mark.parametrize("dimension", [2, 5, 100, 1000, (2, 3), (11, 4)])
def test_get_random_normal_vector(dimension):
    vector = get_random_normal_vector(dimension=dimension)
    if isinstance(dimension, int):
        assert len(vector) == dimension
    elif isinstance(dimension, tuple):
        assert vector.shape == dimension
    assert np.isclose(np.linalg.norm(vector), 1.0)


@pytest.mark.parametrize("dimension", [2, 5, 100, 1000, (2, 3), (11, 4)])
def test_get_random_orthonormal_vector(dimension):
    vector_1 = get_random_normal_vector(dimension=dimension)
    vector_2 = get_random_orthonormal_vector(vector_1)
    if isinstance(dimension, int):
        assert len(vector_1) == len(vector_2) == dimension
    elif isinstance(dimension, tuple):
        assert vector_1.shape == vector_2.shape == dimension
    assert np.isclose(np.linalg.norm(vector_2), 1.0)
    assert np.isclose(np.dot(vector_1.flatten(), vector_2.flatten()), 0.0)


@pytest.mark.parametrize(
    "reference_point,point_to_be_wrapped,period,target_point",
    [
        [np.array([0, 0]), np.array([1, 0]), np.pi, np.array([1, 0])],
        [np.array([0, 0]), np.array([1, 1]), np.pi, np.array([1, 1])],
        [np.array([0, 0]), np.array([1, 0]), 0.8, np.array([0.2, 0])],
        [np.array([0, 0]), np.array([1, 0]), 0.1, np.array([0, 0])],
        [np.array([0, 0]), np.array([1, 0]), 0.11, np.array([0.01, 0])],
        [
            np.array([0, 0]),
            np.array([1, 1]),
            np.sqrt(2) / 2,
            np.array([1 - np.sqrt(2) / 2, 1 - np.sqrt(2) / 2]),
        ],
        [np.array([0, 0]), np.array([3, 4]), 2.5, np.array([0.5, -1])],
        [np.array([1, 1]), np.array([4, 5]), 5, np.array([-1, 0])],
        [
            np.array([[1, 1], [2, 2]]),
            np.array([[4, 5], [6, 7]]),
            3,
            np.array([[1, 2], [3, 1]]),
        ],
        [
            np.array([[1, 1], [2, 2]]),
            np.array([[1.2, -3], [0, 7.1]]),
            1.4,
            np.array([[1.2, 1.2], [1.4, 1.5]]),
        ],
    ],
)
def test_relative_periodic_wrap(
    reference_point, point_to_be_wrapped, period, target_point
):
    resulting_point = relative_periodic_wrap(
        reference_point, point_to_be_wrapped, period
    )
    assert np.allclose(resulting_point, target_point)


@pytest.mark.parametrize(
    "reference_point,trajectory,period,target_trajectory",
    [
        [
            np.array([0, 0]),
            np.array([[4, 0], [3, 0], [2, 0], [1, 0]]),
            np.pi,
            np.array([[4, 0], [3, 0], [2, 0], [1, 0]]),
        ],
        [
            np.array([0, 0]),
            np.array([[6, 0], [5, 0], [4, 0], [3, 0]]),
            2.5,
            np.array([[3.5, 0], [2.5, 0], [1.5, 0], [0.5, 0]]),
        ],
    ],
)
def test_relative_periodic_trajectory_wrap(
    reference_point, trajectory, period, target_trajectory
):
    resulting_trajectory = relative_periodic_trajectory_wrap(
        reference_point, trajectory, period
    )
    assert np.allclose(resulting_trajectory, target_trajectory)


@pytest.mark.parametrize(
    "reference_point,trajectory,period",
    [
        [
            np.array([0, 0]),
            np.array([[4, 0], [3, 0], [2, 0], [1, 0]]),
            0.1,
        ],
    ],
)
def test_relative_periodic_trajectory_wrap_fails_for_wrong_input(
    reference_point, trajectory, period
):
    with pytest.raises(ValueError):
        _ = relative_periodic_trajectory_wrap(reference_point, trajectory, period)


@pytest.mark.parametrize(
    "points,direction,origin,in_units_of_direction,target_points",
    [
        [
            np.array([[0, 0], [1, 1]]),
            np.array([2, 0]),
            np.array([0, 0]),
            False,
            np.array([0, 1]),
        ],
        [
            np.array([[0, 0], [1, 1]]),
            np.array([2, 0]),
            np.array([0, 0]),
            True,
            np.array([0, 0.5]),
        ],
        [
            np.array([[0, 0], [1, 1]]),
            np.array([2, 0]),
            None,
            False,
            np.array([0, 1]),
        ],
        [
            np.array([[0, 0], [1, 1]]),
            np.array([1, 0]),
            np.array([1, 1]),
            False,
            np.array([-1, 0]),
        ],
    ],
)
def test_get_coordinates_on_direction(
    points, direction, origin, in_units_of_direction, target_points
):
    transformed_points = get_coordinates_on_direction(
        points, direction, origin, in_units_of_direction
    )
    assert np.allclose(transformed_points, target_points)


@pytest.mark.parametrize(
    "origin,direction,n_points,endpoints,target_vectors",
    [
        [
            np.array([0, 0]),
            np.array([0, 1]),
            5,
            (-2, 2),
            np.array([[0, -2], [0, -1], [0, 0], [0, 1], [0, 2]]),
        ],
        [
            np.array([1, 1]),
            np.array([-2, 1]),
            4,
            (-1, 1),
            np.array([[3, 0], [5 / 3, 2 / 3], [1 / 3, 4 / 3], [-1, 2]]),
        ],
    ],
)
def test_direction_linspace(origin, direction, n_points, endpoints, target_vectors):
    result = direction_linspace(origin, direction, n_points, endpoints)
    assert np.allclose(result, target_vectors)


@pytest.mark.parametrize(
    "trajectory,n_points,target_trajectory",
    [
        [
            np.array([[1, 0], [2, 0], [3, 0], [4, 0]]),
            4,
            np.array([[1, 0], [2, 0], [3, 0], [4, 0]]),
        ],
        [
            np.array([[1, 0], [2, 0], [3, 0], [7, 0]]),
            4,
            np.array([[1, 0], [3, 0], [5, 0], [7, 0]]),
        ],
        [
            np.array([[0, 0], [3, 0], [3, 3]]),
            5,
            np.array([[0, 0], [1.5, 0], [3, 0], [3, 1.5], [3, 3]]),
        ],
    ],
)
def test_uniformly_distribute_trajectory(trajectory, n_points, target_trajectory):
    transformed_trajectory = uniformly_distribute_trajectory(trajectory, n_points)
    assert np.allclose(transformed_trajectory, target_trajectory)
