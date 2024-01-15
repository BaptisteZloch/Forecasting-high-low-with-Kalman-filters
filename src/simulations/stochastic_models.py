from typing import Union, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from utility.constants import BUSINESS_DAYS


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
        mu: Union[np.float64, int] = 0.2,
        sigma: Union[np.float64, int] = 0.30,
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

    # TODO: Implement the following stochastic models
    @staticmethod
    def simulate_ornstein_uhlenbeck_process():
        pass

    @staticmethod
    def simulate_mean_reverting_process():
        pass

    @staticmethod
    def simulate_heston_model_process():
        pass
