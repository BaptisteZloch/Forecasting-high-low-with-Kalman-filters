from typing import Union
import numpy as np
import numpy.typing as npt

from utility.math_equations import (
    compute_currrent_step_log_price,
    compute_next_step_log_vol,
)


class DSSM:
    @staticmethod
    def f(
        x_k: np.float64,
        u_k: npt.NDArray[np.float64],
        v_k: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64],
    ) -> Union[np.float64, np.float64]:
        """The non linear state function of the DSSM. Predict the next step log volatility (our state variable).

        Args:
            x_k (np.float64): x_k is a 1x1 vector a np.float64 then, with the one element being the log volatility.
            u_k (npt.NDArray[np.float64]): an exogenous/control input 3x1 of the system, assumed known, with the following elements: log price, log price at time t-1, z.
            v_k (npt.NDArray[np.float64]): the process noise that drives the dynamic system,
            w (npt.NDArray[np.float64]): the parameters vector : [kappa, theta, xi, rho, mu, p, ]
        Returns:
            np.float64: _description_
        """
        # TODO: How to handle the v_k noise?

        return compute_next_step_log_vol(
            x_k, u_k[0], u_k[1], u_k[2], w[0], w[1], w[2], w[3], w[4], w[5], dt=1
        )

    @staticmethod
    def h(x_k, u_k, v_k, w) -> np.float64:
        return compute_currrent_step_log_price(x_k, u_k[0], w[4], u_k[2], u_k[1])
