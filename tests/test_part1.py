import numpy as np

from part1.ols_implementation import hat_matrix, ols_fit


def test_ols_fit_perfect_line():
    X = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    y = np.array([1.0, 3.0, 5.0, 7.0])

    beta_hat, sigma2 = ols_fit(X, y)

    np.testing.assert_allclose(beta_hat, np.array([1.0, 2.0]), atol=1e-10)
    assert np.isclose(sigma2, 0.0, atol=1e-10)


def test_hat_matrix_properties():
    X = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )

    H = hat_matrix(X)

    assert H.shape == (4, 4)
    np.testing.assert_allclose(H.T, H, atol=1e-10)
    np.testing.assert_allclose(H @ H, H, atol=1e-10)
