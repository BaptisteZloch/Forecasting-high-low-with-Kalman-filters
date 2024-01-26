from typing import Union, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from utility.constants import BUSINESS_DAYS
from utility.math_equations import compute_current_price, compute_current_variance


class StochasticModels:
    @staticmethod
    def simulate_random_walk_process(
        T: Union[np.float64, int] = 2,
        s0: Union[np.float64, int] = 100,
    ):
        """Simulate a random walk from a Brownian motion for daily data.

        Args:
            T (Union[np.float64, int], optional): The horizon in years. Defaults to 2.
            s0 (Union[np.float64, int], optional): The starting point value. Defaults to 100.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]: A tuple containing the time and the corresponding path.
        """
        # Number of steps to simulate
        n_steps = int(T * BUSINESS_DAYS)
        # Generate synthetic stock price data using random walk
        # Start from a stock price of S0
        t = np.linspace(0, T, n_steps + 1)
        return t, np.cumsum(np.random.normal(0, T, n_steps + 1)) + s0

    @staticmethod
    def simulate_arithmetic_brownian_motion_process(
        T: Union[np.float64, int] = 2,
        mu: np.float64 = 0.2,
        sigma: np.float64 = 0.30,
        s0: Union[np.float64, int] = 100,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Simulate an arithmetic Brownian motion for daily data.

        Args:
            T (Union[np.float64, int], optional): The horizon in years. Defaults to 2.
            mu (Union[np.float64, int], optional): The average growth rate by year. Defaults to 0.2.
            sigma (Union[np.float64, int], optional): The annual volatility. Defaults to 0.30.
            s0 (Union[np.float64, int], optional): The starting point value. Defaults to 100.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]: A tuple containing the time and the corresponding path.
        """
        # Number of steps to simulate
        n_steps = int(T * BUSINESS_DAYS)
        # Standard Brownian motion
        w_t = np.random.normal(0, T, size=n_steps)

        dt = T / n_steps
        path = np.ones((n_steps + 1))
        path[0] = s0
        path[1:] = (
            1 + (mu / BUSINESS_DAYS) * dt + (sigma / (BUSINESS_DAYS**0.5)) * w_t
        )
        path = np.cumprod(path)
        t = np.linspace(0, T, n_steps + 1)
        return t, path

    @staticmethod
    def simulate_geometric_brownian_motion_process(
        T: Union[np.float64, int] = 2,
        mu: Union[np.float64, int] = 0.20,
        sigma: Union[np.float64, int] = 0.40,
        s0: Union[np.float64, int] = 100,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Simulate a geometric Brownian motion for daily data.

        Args:
            T (Union[np.float64, int], optional): The horizon in years. Defaults to 2.
            mu (Union[np.float64, int], optional): The average growth rate by year. Defaults to 0.2.
            sigma (Union[np.float64, int], optional): The annual volatility. Defaults to 0.30.
            s0 (Union[np.float64, int], optional): The starting point value. Defaults to 100.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]: A tuple containing the time and the corresponding path.
        """
        # Number of steps to simulate
        n_steps = int(T * BUSINESS_DAYS)
        # Standard Brownian motion
        w_t = np.random.normal(0, T, size=n_steps)

        dt = T / n_steps
        path = np.ones((n_steps + 1))
        path[0] = s0

        for j in tqdm(
            range(n_steps),
            desc="Simulating path...",
            leave=False,
        ):
            path[j + 1] = path[j] + (
                (mu / BUSINESS_DAYS) * path[j] * dt
                + (sigma / (BUSINESS_DAYS**0.5)) * path[j] * w_t[j]
            )
        t = np.linspace(0, T, n_steps + 1)
        return t, path

    @staticmethod
    def simulate_heston_process_with_jump(
        T: float = 2,
        s0: float = 100,
        kappa: float = 4,
        theta: float = 0.27,
        xi: float = 0.99,
        rho: float = -0.2,
        mu: float = 0.3,
        lambda_0: float = 0.01,
        mu_jump: float = -0.05,
        sigma_jump: float = 0.1,
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Generate a Heston model path with jumps.

        Args:
        -----
            T (int, optional): The horizon in years. Defaults to 2.
            s0 (float, optional): The starting point value. Defaults to 100.
            kappa (float, optional): Mean reversion speed. Defaults to 4.
            theta (float, optional): Long-run mean of variance. Defaults to 0.27.
            xi (float, optional): Volatility of volatility. Defaults to 0.99.
            rho (float, optional): Correlation between asset price and variance. Defaults to -0.2.
            mu (float, optional): The average growth rate by year. Defaults to 0.3.
            lambda_0 (float, optional): Intensity of jumps. Defaults to 0.01.
            mu_jump (float, optional): Mean jump size. Defaults to -0.05.
            sigma_jump (float, optional): Standard deviation of jump size. Defaults to 0.1.

        Returns:
        -----
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: The time, the asset price and the volatility paths.
        """
        # Initialize variables
        n_steps = int(252 * T)
        dt = T / n_steps
        S = np.zeros(n_steps + 1)
        V = np.zeros(n_steps + 1)

        # Generate correlated Brownian motions
        dW = np.random.multivariate_normal(
            [0, 0], [[dt, rho * dt], [rho * dt, dt]], size=(n_steps + 1,)
        )

        # Generate Poisson process for jumps
        poisson_jumps = np.random.poisson(lambda_0, n_steps + 1) * np.random.normal(
            mu_jump, sigma_jump, n_steps + 1
        )

        # Generate asset price and volatility paths
        S[0] = s0
        V[0] = theta
        for i in tqdm(
            range(1, n_steps + 1), desc="Generating path", total=n_steps, leave=False
        ):
            V[i] = (
                V[i - 1]
                + (
                    kappa * (theta - V[i - 1]) * dt
                    + xi * np.sqrt(np.abs(V[i - 1])) * dW[i, 0]
                )
                + np.abs(poisson_jumps[i])
            )
            S[i] = S[i - 1] * (
                np.exp(
                    (mu - 0.5 * V[i - 1]) * dt + np.sqrt(np.abs(V[i - 1])) * dW[i, 1]
                )
                + poisson_jumps[i]
            )
        t = np.linspace(0, T, n_steps + 1)
        return t, S, V

    @staticmethod
    def simulate_heston_process(
        T: float = 2,
        s0: float = 100,
        kappa: float = 19,
        theta: float = 0.27**2,
        xi: float = 0.01,
        rho: float = -0.2,
        mu: float = 0.3,
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Generate a Heston model path with jumps.

        Args:
        -----
            T (int, optional): The horizon in years. Defaults to 2.
            s0 (float, optional): The starting point value. Defaults to 100.
            kappa (float, optional): Mean reversion speed. Defaults to 4.
            theta (float, optional): Long-run mean of variance. Defaults to 0.27.
            xi (float, optional): Volatility of volatility. Defaults to 0.99.
            rho (float, optional): Correlation between asset price and variance. Defaults to -0.2.
            mu (float, optional): The average growth rate by year. Defaults to 0.3.


        Returns:
        -----
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: The time, the asset price and the volatility paths.
        """
        assert 2 * kappa * theta > xi**2, "Feller condition not satisfied"
        # Initialize variables
        n_steps = int(252 * T)
        dt = T / n_steps
        S = np.zeros(n_steps + 1)
        V = np.zeros(n_steps + 1)

        # Generate correlated Brownian motions
        dW = np.random.multivariate_normal(
            [0, 0], [[dt, rho * dt], [rho * dt, dt]], size=(n_steps + 1,)
        )

        # Generate asset price and volatility paths
        S[0] = s0
        V[0] = theta
        for i in tqdm(
            range(1, n_steps + 1), desc="Generating path", total=n_steps, leave=False
        ):
            V[i] = compute_current_variance(V[i - 1], kappa, theta, xi, dt, dW[i, 0])
            S[i] = compute_current_price(S[i - 1], V[i - 1], mu, dt, dW[i, 1])

        t = np.linspace(0, T, n_steps + 1)
        return t, S, V

    @staticmethod
    def generate_two_correlated_brownian(
        rho: float = 0.5, n_steps: int = 1000, dt: float = 0.01
    ):
        """Generate two correlated Brownian motions.

        Args:
            rho (float, optional): Correlation between the two Brownian motions. Defaults to 0.5.
            n_steps (int, optional): Number of steps to simulate. Defaults to 1000.
            dt (float, optional): Time step. Defaults to 0.01.

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: The two correlated Brownian motions.
        """
        # Generate correlated Brownian motions
        dW = np.random.multivariate_normal(
            [0, 0], [[dt, rho * dt], [rho * dt, dt]], size=(n_steps + 1,)
        )
        return dW[:, 0], dW[:, 1]

    # TODO: Implement the following stochastic models
    @staticmethod
    def simulate_mean_reverting_process():
        raise NotImplementedError

    @staticmethod
    def simulate_heston_model_process():
        raise NotImplementedError
