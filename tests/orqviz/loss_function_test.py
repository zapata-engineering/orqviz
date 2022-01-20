import numpy as np

from orqviz.loss_function import LossFunctionWrapper


def mock_function(params, b, c):
    return np.sum(params * b + c)


class TestLossFunctionWrapper:
    def test_if_wraps_args(self):
        b = 10
        c = 2
        params = np.array([1, 2, 3, 4])
        target_value = mock_function(params, b, c)

        loss_function = LossFunctionWrapper(mock_function, b, c)

        result = loss_function(params=params)
        assert result == target_value

    def test_if_wraps_kwargs(self):
        b = 10
        c = 2
        params = np.array([1, 2, 3, 4])
        target_value = mock_function(params, b, c)

        loss_function = LossFunctionWrapper(mock_function, b=b, c=c)

        result = loss_function(params=params)
        assert result == target_value

    def test_if_wraps_both_args_and_kwargs(self):
        b = 10
        c = 2
        params = np.array([1, 2, 3, 4])
        target_value = mock_function(params, b, c)

        loss_function = LossFunctionWrapper(mock_function, b, c=c)

        result = loss_function(params=params)

        assert result == target_value

    def test_tracks_average_call_time(self):
        b = 10
        c = 2
        call_count = 10

        loss_function = LossFunctionWrapper(mock_function, b, c=c)
        for i in range(call_count):
            _ = loss_function(params=np.random.random(i))

        assert loss_function.average_call_time is not None

    def test_tracks_call_count(self):
        b = 10
        c = 2
        call_count = 10

        loss_function = LossFunctionWrapper(mock_function, b, c=c)
        for i in range(call_count):
            _ = loss_function(params=np.random.random(i))

        assert loss_function.call_count == call_count

    def test_tracks_min_value(self):
        b = 10
        c = 2
        call_count = 10

        loss_function = LossFunctionWrapper(mock_function, b, c=c)
        min_value = np.inf
        for i in range(call_count):
            value = loss_function(params=np.random.random(i))
            if value < min_value:
                min_value = value

        assert loss_function.min_value == min_value

    def test_reset(self):
        b = 10
        c = 2
        call_count = 10

        loss_function = LossFunctionWrapper(mock_function, b, c=c)
        for i in range(call_count):
            _ = loss_function(params=np.random.random(i))

        loss_function.reset()

        assert loss_function.average_call_time is None
        assert loss_function.call_count == 0
        assert loss_function.min_value is None
