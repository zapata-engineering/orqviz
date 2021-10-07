from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np

from ..aliases import ParameterVector


@dataclass()
class HessianEigenobject:
    """Data structure for Hessian matrix.
        Eigenvalues and Eigenvectors are automatically calculated and sorted.

    Args:
        params: Parameter vector at which the Hessian matrix was measured
        hessian_matrix: Hessian matrix evaluated at params as a 2D numpy array.
    """

    params: ParameterVector
    hessian_matrix: np.ndarray
    eigenvectors: List[np.ndarray] = field(init=False)
    eigenvalues: np.ndarray = field(init=False)

    def __post_init__(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.hessian_matrix)
        eigenvectors = eigenvectors.T
        sorted_ind = np.argsort(eigenvalues)
        self.eigenvalues = eigenvalues[sorted_ind]
        self.eigenvectors = eigenvectors[sorted_ind]
