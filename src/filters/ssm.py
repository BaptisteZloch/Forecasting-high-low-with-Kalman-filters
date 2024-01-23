from typing import List, Union
import numpy as np
import numpy.typing as npt

from utility.math_equations import (
    compute_currrent_step_log_price,
    compute_next_step_log_vol,
)


class DSSM:
    # _instance = None

    def __init__(
        self,
        w_params: Union[List[np.float64], npt.NDArray[np.float64]],
    ) -> None:
        """The constructor of the DSSM class.

        Args:
        ----
            w_params (Union[List[np.float64], npt.NDArray[np.float64]]): the parameters vector : [kappa, theta, xi, rho, mu, p, ]
        """
        self.w = w_params

    def f(
        self,
        x_k: npt.NDArray[np.float64],
        u_k: npt.NDArray[np.float64],
        v_k: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """The non linear state function of the DSSM. Predict the next step log volatility (our state variable).

        Args:
        ----
            x_k (npt.NDArray[np.float64]): x_k is the state vector of the system, with the following elements: log vol at time t, log vol at time t-1, log vol at time t-2.
            u_k (npt.NDArray[np.float64]): an exogenous/control input 3x1 of the system, assumed known, with the following elements: log price, log price at time t-1, z.
            v_k (npt.NDArray[np.float64]): the process noise that drives the dynamic system

        Returns:
        ----
            np.float64: _description_
        """
        # TODO: How to handle the v_k noise?

        return np.array(
            [
                np.abs(compute_next_step_log_vol(
                    x_k[0],  # log vol at time t
                    x_k[1],  # log vol at time t-1
                    x_k[2],  # log vol at time t-2
                    u_k[0],  # brownian motion at time t
                    self.w[0],  # kappa
                    self.w[1],  # theta
                    self.w[2],  # xi
                    self.w[3],  # rho
                    self.w[4],  # mu
                    self.w[5],  # p
                    dt=1,
                )),  # log vol at time t+1
                x_k[1],  # log price at time t
                x_k[2],  # log price at time t-1
            ]
        )

    def h(
        self,
        x_k: npt.NDArray[np.float64],
        u_k: npt.NDArray[np.float64],
        v_k: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [x_k[0], compute_currrent_step_log_price(x_k[1], x_k[0], self.w[4], u_k[1], dt=1), x_k[1]]
        )
        # return np.array(
        #     [compute_currrent_step_log_price(x_k[1], x_k[0], self.w[4], u_k[1], dt=1)]
        # )

    # @property
    # def w(self) -> Union[List[np.float64], npt.NDArray[np.float64]]:
    #     return self.w

    # @w.setter
    # def w(self, w_params: Union[List[np.float64], npt.NDArray[np.float64]]) -> None:
    #     self.w = w_params

    # def __new__(cls, *args, **kwargs) -> Self:
    #     if cls._instance is None:
    #         cls._instance = super(DSSM, cls).__new__(cls)
    #     return cls._instance
