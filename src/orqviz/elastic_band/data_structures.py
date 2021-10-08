from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from scipy.interpolate import interp1d

from ..aliases import ArrayOfParameterVectors, Weights
from ..scans import eval_points_on_path


class Chain(NamedTuple):
    """Data structure for Chain in the Nudged Elastic Band (NEB) algorithm.
        This is how we call the NEB with pivot points.

    Args:
        pivots: Array of parameter vectors which form
            a piece-wise linearly connected chain
    """

    pivots: ArrayOfParameterVectors

    def get_weights(self) -> Weights:
        chain_weights = np.linalg.norm(np.diff(self.pivots, axis=0), axis=1)
        chain_weights /= np.sum(chain_weights)
        cum_weights = np.cumsum(chain_weights)
        matching_cum_weights = np.insert(cum_weights, 0, 0)
        matching_cum_weights[-1] = 1
        return matching_cum_weights

    def evaluate_on_pivots(self, loss_function: Callable) -> np.ndarray:
        return eval_points_on_path(self.pivots, loss_function)

    @property
    def n_pivots(self) -> int:
        return len(self.pivots)

    @property
    def n_params(self) -> int:
        return len(self.pivots[0])


class ChainPath(NamedTuple):
    """Data structure for the piece-wise linear path that is defined by a Chain.
    Args:
        Chain
    """

    primary_chain: Chain

    @property
    def primary_weights(self) -> Weights:
        return self.primary_chain.get_weights()

    def generate_chain(self, n_points: int) -> Chain:
        weight_interpolator = interp1d(
            np.linspace(0, 1, num=len(self.primary_weights)),
            self.primary_weights,
        )
        weights = weight_interpolator(np.linspace(0, 1, num=n_points))
        return self._get_chain_from_weights(weights)

    def generate_uniform_chain(self, n_points: int) -> Chain:
        weights = np.linspace(0, 1, num=n_points)
        return self._get_chain_from_weights(weights)

    def evaluate_points_on_path(
        self, n_points: int, loss_function: Callable, weighted: bool = False
    ) -> np.ndarray:
        if weighted:
            chain = self.generate_chain(n_points)
        else:
            chain = self.generate_uniform_chain(n_points)
        return chain.evaluate_on_pivots(loss_function)

    def _get_chain_from_weights(self, weights: Weights) -> Chain:
        chain_diff = np.cumsum(
            np.linalg.norm(np.diff(self.primary_chain.pivots, axis=0), axis=1)
        )
        chain_diff /= max(chain_diff)
        chain_diff = np.insert(chain_diff, 0, 0)

        interpolator = interp1d(
            chain_diff, self.primary_chain.pivots, kind="linear", axis=0
        )
        return Chain(interpolator(weights))
