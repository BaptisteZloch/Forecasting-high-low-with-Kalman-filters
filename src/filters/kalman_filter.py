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


# https://towardsdatascience.com/exposing-the-power-of-the-kalman-filter-1b78621c3f56
class KalmanFilterND:
    """
    An implementation of the classic Kalman Filter for linear dynamic systems.
    The Kalman Filter is an optimal recursive data processing algorithm which
    aims to estimate the state of a system from noisy observations.

    Attributes:
        F (npt.NDArray[np.float64]): The state transition matrix.
        B (npt.NDArray[np.float64]): The control input marix.
        H (npt.NDArray[np.float64]): The observation matrix.
        u (npt.NDArray[np.float64]): the control input.
        Q (npt.NDArray[np.float64]): The process noise covariance matrix.
        R (npt.NDArray[np.float64]): The measurement noise covariance matrix.
        x (npt.NDArray[np.float64]): The mean state estimate of the previous step (k-1).
        P (npt.NDArray[np.float64]): The state covariance of previous step (k-1).
    """

    def __init__(
        self,
        F: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
        H: npt.NDArray[np.float64],
        Q: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
        x0: npt.NDArray[np.float64],
        P0: npt.NDArray[np.float64],
    ):
        """
        Initializes the Kalman Filter with the necessary matrices and initial state.

        Args:
        ----
            F (npt.NDArray[np.float64]): The state transition matrix.
            B (npt.NDArray[np.float64]): The control input marix.
            H (npt.NDArray[np.float64]): The observation matrix.
            u (npt.NDArray[np.float64]): the control input.
            Q (npt.NDArray[np.float64]): The process noise covariance matrix.
            R (npt.NDArray[np.float64]): The measurement noise covariance matrix.
            x0 (npt.NDArray[np.float64]): The initial state estimate.
            P0 (npt.NDArray[np.float64]): The initial state covariance matrix.
        """
        self.F = F  # State transition matrix
        self.B = B  # Control input matrix
        self.u = u  # Control vector
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial estimate covariance

    def predict(self) -> npt.NDArray[np.float64]:
        """
        Predicts the state and the state covariance for the next time step.
        """
        self.x = self.F @ self.x + self.B @ self.u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        Updates the state estimate with the latest measurement.

        Parameters:
            z (npt.NDArray[np.float64]): The measurement at the current step.
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.x  # Challenges with Non-linear Systems
