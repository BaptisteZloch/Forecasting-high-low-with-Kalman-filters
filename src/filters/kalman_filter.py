from typing import Tuple, Union
import numpy as np
import numpy.typing as npt

from tqdm import tqdm


# from : https://medium.com/@akjha22/kalman-filters-for-stock-price-signal-generation-f64015da637d
class SimplestKalmanFilter1D:
    def __init__(self, Q: Union[float, int], R: Union[float, int]):
        # Process noise variance
        self.Q = Q
        # Measurement noise variance
        self.R = R

    def __before_estimation_init(self, z: npt.NDArray[np.float64]):
        # Number of epochs
        self.num_days = z.shape[0]
        # Initial state estimate
        self.P_hat = np.zeros(self.num_days)
        # Initial state estimate error variance
        self.P_var = np.zeros(self.num_days)

    def fit_predict(
        self, z: npt.NDArray[np.float64], verbose: bool = False
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self.__before_estimation_init(z)

        # Initial estimate of state
        self.P_hat[0] = z[0]

        # Initial estimate of state variance
        self.P_var[0] = 1.0

        for t in tqdm(
            range(1, self.num_days),
            desc="Kalman Filter Progress",
            leave=True,
            total=self.num_days - 1,
        ):
            # ******* Prediction Step *******

            # Predicted state estimate
            P_hat_minus = self.P_hat[t - 1]
            # Predicted error variance
            P_var_minus = self.P_var[t - 1] + self.Q

            # ******* Update Step *******

            # Kalman gain
            Kt = P_var_minus / (P_var_minus + self.R)

            if verbose is True:
                print(f"Kalman Gain at epoch {t} : {Kt:.4f}")

            # Updated state estimate
            self.P_hat[t] = P_hat_minus + Kt * (z[t] - P_hat_minus)
            # Updated estimate of state variance
            self.P_var[t] = (1 - Kt) * P_var_minus

        return self.P_hat, self.P_var
