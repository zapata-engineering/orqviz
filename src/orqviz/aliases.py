from typing import Callable

import numpy as np

"""
ParameterVector, ArrayOfParameterVectors, GridOfParameterVectors
are all aliases of numpy.ndarray and are used throughout this library.
They indicate whether the numpy.ndarray that is expected by a method or class
is a 1D, 2D, or 3D numpy.ndarray, respectively. The last dimension, for ParameterVector
the only dimension, is always of size number_of_parameters, while the other dimensions
indicate how many of them there are.
"""
ParameterVector = np.ndarray  # ND array
ArrayOfParameterVectors = np.ndarray  # Array of ND arrays
GridOfParameterVectors = np.ndarray  # Grid of ND arrays
Weights = np.ndarray  # 1D vector of floats from 0-1
DirectionVector = np.ndarray  # ND array with same shape as ParameterVector
LossFunction = Callable[
    [ParameterVector], float
]  # Function that can be scanned with orqviz
GradientFunction = Callable[
    [ParameterVector, DirectionVector], float
]  # Returns partial derrivative of LossFunction wrt DirectionVector
FullGradientFunction = Callable[
    [ParameterVector], np.ndarray
]  # Returns all partial derrivatives of LossFunction wrt each parameter
