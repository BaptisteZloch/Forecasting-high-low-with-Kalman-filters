from typing import Literal
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def generate_brownian_paths(
    n_paths: int,
    n_steps: int,
    T: float | int,
    mu: float | int,
    sigma: float | int,
    s0: float | int,
    brownian_type: Literal["ABM", "GBM"] = "ABM",
    get_time: bool = True,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = np.log(s0)

    for i in tqdm(
        range(0, n_paths), desc="Simulating Brownian Motion paths...", leave=False
    ):
        for j in tqdm(
            range(n_steps),
            desc="Simulating Brownian Motion path's steps...",
            leave=False,
        ):
            paths[i, j + 1] = (
                paths[i, j]
                + (mu * 0.5 * sigma**2) * dt
                + sigma * np.random.normal(0, np.sqrt(dt))
            )

    if brownian_type == "GBM":
        paths = np.exp(paths)

    return np.linspace(0, T, n_steps + 1), paths if get_time is True else paths
