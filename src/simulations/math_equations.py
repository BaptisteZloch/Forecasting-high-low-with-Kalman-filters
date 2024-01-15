from math import exp, log, sqrt


def compute_next_step_log_vol(
    current_step_log_vol: float,
    previous_step_price: float,
    previous_previous_step_price: float,
    current_step_z: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    mu: float,
    p: float,
    dt: int = 1,
) -> float:
    """Compute the log volatility at time t + dt.

    Args:
        current_step_log_vol (float):  The log volatility at time t.
        previous_step_price (float): _description_
        previous_previous_step_price (float): _description_
        current_step_z (float): _description_
        kappa (float):  The mean reversion speed.
        theta (float): The mean reversion level.
        xi (float): The volatility of volatility.
        rho (float): The correlation between the log volatility and the Brownian motion.
        mu (float): The drift of the log price.
        p (float): The power of the log volatility.
        dt (int, optional): The time step. Defaults to 1.

    Returns:
        float: The log volatility at time t + dt.
    """
    current_step_vol = exp(current_step_log_vol)
    return (
        current_step_log_vol
        + (1 / current_step_vol)
        * (
            kappa * (theta - current_step_vol)
            - 0.5 * xi**2 * current_step_vol ** (2 * p - 1)
            - rho * xi * current_step_vol ** (p - 0.5) * (mu - 0.5 * current_step_vol)
        )
        * dt
        + rho
        * xi
        * current_step_vol ** (p - (3 / 2))
        * (log(previous_step_price) - log(previous_previous_step_price))
        + xi * current_step_vol ** (p - 1) * sqrt(dt) * sqrt(1 - rho) * current_step_z
    )


def compute_currrent_step_log_price(
    previous_step_log_price: float,
    current_step_vol: float,
    mu: float,
    current_step_b: float,
    dt: int = 1,
) -> float:
    """Compute the current log price at time t.

    Args:
        previous_step_log_price (float): The log price at time t - dt.
        current_step_vol (float): The log volatility at time t.
        mu (float): _description_
        current_step_b (float): _description_
        dt (float): The time step.

    Returns:
        float: The log price at time t.
    """
    return (
        previous_step_log_price
        + (mu - 0.5 * current_step_vol) * dt
        + sqrt(dt) * sqrt(current_step_vol) * current_step_b
    )
