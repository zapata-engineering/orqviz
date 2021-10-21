from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .aliases import ArrayOfParameterVectors, ParameterVector


def get_random_normal_vector(dimension: int) -> ParameterVector:
    """Helper function to generate a vector with a specified dimension and norm=1."""
    random_vector = np.random.normal(0, 1, size=dimension)
    return random_vector / np.linalg.norm(random_vector)


def get_random_orthonormal_vector(base_vector: ParameterVector) -> ParameterVector:
    """Helper function to generate a random orthogonal vector with respect to
    a provided base vector."""
    random_vector = np.random.normal(size=base_vector.shape)
    new_vector = (
        random_vector
        - np.dot(random_vector, base_vector)
        * base_vector
        / np.linalg.norm(base_vector) ** 2
    )
    return new_vector / np.linalg.norm(new_vector)


def relative_periodic_wrap(
    reference_point: ParameterVector,
    point_to_be_wrapped: ParameterVector,
    period: float = 2 * np.pi,
) -> ParameterVector:
    """Function that returns a wrapped 'copy' of a point to be wrapped such that
        the distance between it and the reference point is minimal inside
        the specified period.

    Args:
        reference_point: Reference point for periodic wrapping of the second point.
        point_to_be_wrapped: Point that is wrapped to a copy of itself such that
            its distance to the reference point is minimal inside the specified period.
        period: Periodicity of each parameter of the the point that is to be wrapped.
            Defaults to 2*np.pi.
    """
    option1 = (reference_point - point_to_be_wrapped) % period
    option2 = (point_to_be_wrapped - reference_point) % period
    diff = np.where(option1 > option2, option2, -option1)
    wrapped_point = reference_point + diff
    return wrapped_point


def relative_periodic_trajectory_wrap(
    reference_point: ParameterVector,
    trajectory: ArrayOfParameterVectors,
    period: float = 2 * np.pi,
) -> ArrayOfParameterVectors:
    """Function that returns a wrapped 'copy' of a parameter trajectory such that
        the distance between the final point of the trajectory and the reference point
        is minimal inside the specified period.
        The rest of the trajectory is being transformed in the same manner.
        NOTE:
            It only works as intended if the period is larger than the distance
            between the consecutive points in the trajectory.

    Args:
        reference_point: Reference point for periodic wrapping of the trajectory.
        trajectory: Trajectory that is wrapped to a copy of itself such that
            the distance between the final point in the trajectory
            and the reference point is minimal.
        period: Periodicity of each parameter in each point of the trajectory.
            Defaults to 2*np.pi.
    """
    if not np.all(np.linalg.norm(np.diff(trajectory, axis=0), axis=1) < period):
        raise ValueError(
            "Distances between consecutive points must be smaller than period."
        )
    wrapped_trajectory = np.copy(trajectory).astype(float)
    wrapped_trajectory[-1] = relative_periodic_wrap(
        reference_point, trajectory[-1], period=period
    )
    for ii in range(2, len(wrapped_trajectory) + 1):
        wrapped_trajectory[-ii] = relative_periodic_wrap(
            wrapped_trajectory[-ii + 1], trajectory[-ii], period=period
        )

    return wrapped_trajectory


def get_coordinates_on_direction(
    points: ArrayOfParameterVectors,
    direction: np.ndarray,
    origin: Optional[ParameterVector] = None,
    in_units_of_direction: bool = False,
) -> np.ndarray:
    """Helper function to calculate the projection of points onto a direction
        to extract the coordinates in that direction.

    Args:
        points: Points to be projected on the direction vector.
        direction: Direction vector to project the points on.
        origin: Origin for the projection. Defaults to None.
        in_units_of_direction: Flag to indicate whether to return coordinates in units
            of the direction vector.
            If False, returns coordinates in euclidean distances. Defaults to False.
    """
    if origin is not None:
        points = points - origin
    norm_direction = np.linalg.norm(direction)
    if in_units_of_direction:
        direction = direction / norm_direction
    return np.dot(points, direction) / norm_direction


def direction_linspace(
    origin: ParameterVector,
    direction: np.ndarray,
    n_points: int,
    endpoints: Tuple[float, float] = (-1, 1),
) -> ArrayOfParameterVectors:
    """Helper function to wrap np.linspace in order to create points on a specified
    direction around an origin."""
    return np.linspace(
        origin + endpoints[0] * direction,
        origin + endpoints[1] * direction,
        num=n_points,
    )


def uniformly_distribute_trajectory(
    parameter_trajectory: ArrayOfParameterVectors,
    n_points: int,
) -> ArrayOfParameterVectors:
    """Function to distribute points uniformly (in euclidean distance) along a path
    that is given by a parameter trajectory, i.e. an array of parameter vectors.
    Returns an array of parameter vectors where the first and last entries match
    those of the passed parameter trajectory and all entries are equally distant
    in euclidean distance.
    """
    trajectory_weights = np.linalg.norm(np.diff(parameter_trajectory, axis=0), axis=1)
    trajectory_weights /= np.sum(trajectory_weights)
    cum_weights = np.cumsum(trajectory_weights)
    cum_weights = np.insert(cum_weights, 0, 0)
    cum_weights[-1] = 1  # for stability insert an integer 1

    weight_interpolator = interp1d(
        cum_weights,
        parameter_trajectory,
        axis=0,
    )
    eval_points = np.linspace(0, 1, num=n_points)
    return weight_interpolator(eval_points)
