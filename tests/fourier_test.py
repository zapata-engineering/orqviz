import orqviz
import numpy as np


def test_inverse():
    random = np.random.uniform(-np.pi, np.pi, size=20)
    transformed = orqviz.dct.dct(random)
    inversed = orqviz.dct.dct_inverse(transformed)

    np.testing.assert_array_almost_equal(random, inversed)