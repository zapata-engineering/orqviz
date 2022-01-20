import time
from typing import Callable, Optional

import numpy as np


def _calculate_new_average(
    previous_average: Optional[float], count: int, new_value: float
) -> float:
    if previous_average is None:
        return new_value
    else:
        return (count * previous_average + new_value) / (count + 1)


class LossFunctionWrapper:
    def __init__(self, loss_function: Callable, *args, **kwargs):
        def wrapped_loss_function(params: np.ndarray) -> float:
            return loss_function(params, *args, **kwargs)

        self.loss_function = wrapped_loss_function
        self.call_count = 0
        self.average_call_time: Optional[float] = None
        self.min_value: Optional[float] = None

    def __call__(self, params: np.ndarray) -> float:
        start_time = time.perf_counter()
        value = self.loss_function(params)
        total_time = time.perf_counter() - start_time
        self.average_call_time = _calculate_new_average(
            self.average_call_time, self.call_count, total_time
        )
        self.call_count += 1
        if self.min_value is None or value < self.min_value:
            self.min_value = value
        return value

    def reset(self) -> None:
        self.call_count = 0
        self.average_call_time = None
        self.min_value = None
