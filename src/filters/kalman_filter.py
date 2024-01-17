from typing import Tuple, Union, Optional
import numpy as np
import numpy.typing as npt

from tqdm import tqdm


# from : https://medium.com/@akjha22/kalman-filters-for-stock-price-signal-generation-f64015da637d
class SimplestKalmanFilter1D:
    def __init__(self, Q: Union[np.float64, int], R: Union[np.float64, int]):
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


# Adapted from : https://towardsdatascience.com/exposing-the-power-of-the-kalman-filter-1b78621c3f56 and https://codingcorner.org/unscented-kalman-filter-ukf-best-explanation-with-python/
class LinearKalmanFilterND:
    """

    Attributes:
    ----
        F (npt.NDArray[np.float64]): The state transition matrix
        H (npt.NDArray[np.float64]): The observation matrix
        Q (npt.NDArray[np.float64]): The process noise variance
        R (npt.NDArray[np.float64]): The measurement noise variance
        B (Optional[npt.NDArray[np.float64]], optional): The Control matrix. Defaults to None.
        u (Optional[npt.NDArray[np.float64]], optional): The Control vector. Defaults to None.
        x_hat (npt.NDArray[np.float64]): The state estimate
        P_var (npt.NDArray[np.float64]): The state estimate error variance
        y_hat (npt.NDArray[np.float64]): The measure estimate
        V_var (npt.NDArray[np.float64]): The measure estimate error variance

    Methods:
    ----
        Public:
            fit_predict(z: npt.NDArray[np.float64], verbose: bool = False) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: The Kalman Filter fit and predict method
        Private:
            __KF_predict(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], P: npt.NDArray[np.float64], V: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: The Kalman Filter prediction step
            __KF_correct(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], y_obs: npt.NDArray[np.float64], P: npt.NDArray[np.float64], V: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: The Kalman Filter correction step

    Examples:
    ----
    1D Kalman Filter implementation example
    ```py
    kf_matriciel = SimplestKalmanFilter1D(
        F=np.array([[1.0]]),
        H=np.array([[1.0]]),
        Q=np.array([[0.5]]),
        R=np.array([[0.5]]),
        # B = np.array([[0.0]]),
        # u = np.array([[1.0]])
    )

    x_hat, P_var, y_hat, V_var = kf_matriciel.fit_predict(
        df.prices.to_numpy().reshape(-1, 1), verbose=False
    )
    ```

    2D Kalman Filter implementation example
    ```py
    kf_matriciel = SimplestKalmanFilter1D(
        F=np.array([[1.0,0.0],[0.0,1.0]]),
        H=np.array([[1.0,0.0],[0.0,1.0]]),
        Q=np.array([[1.0,0.0],[0.0,1.0]]),
        R=np.array([[1.0,0.0],[0.0,1.0]]),
        # B = np.array([[0.0]]),
        # u = np.array([[1.0]])
    )

    x_hat, P_var, y_hat, V_var = kf_matriciel.fit_predict(
        df[['prices','vol']].to_numpy(), verbose=False
    )
    ```
    """

    def __init__(
        self,
        F: npt.NDArray[np.float64],
        H: npt.NDArray[np.float64],
        Q: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
        B: Optional[npt.NDArray[np.float64]] = None,
        u: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """The constructor of the Simple linear Kalman Filter

        Args:
            F (npt.NDArray[np.float64]): The state transition matrix
            H (npt.NDArray[np.float64]): The observation matrix
            Q (npt.NDArray[np.float64]): The process noise variance
            R (npt.NDArray[np.float64]): The measurement noise variance
            B (Optional[npt.NDArray[np.float64]], optional): The Control matrix. Defaults to None.
            u (Optional[npt.NDArray[np.float64]], optional): The Control vector. Defaults to None.
        """

        # State transition matrix
        self.F = F
        # Observation matrix
        self.H = H
        # Process noise variance
        self.Q = Q
        # Measurement noise variance
        self.R = R
        if B is None:
            B = np.zeros((self.F.shape[0], self.F.shape[-1]))
        if u is None:
            u = np.ones((self.F.shape[-1], 1))
        # Control matrix
        self.B = B
        # Control vector
        self.u = u

    def __before_estimation_init(self, z: npt.NDArray[np.float64]):
        # Initial state estimate
        self.x_hat = np.zeros(shape=(z.shape[0], z.shape[-1]))
        # Initial state estimate error variance
        self.P_var = np.ones(shape=(z.shape[0], z.shape[-1], z.shape[-1]))
        # Initial measure estimate
        self.y_hat = np.zeros(shape=(z.shape[0], z.shape[-1]))
        # Initial measure estimate error variance
        self.V_var = np.ones(shape=(z.shape[0], z.shape[-1], z.shape[-1]))

    def fit_predict(
        self, z: npt.NDArray[np.float64], verbose: bool = False
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        self.__before_estimation_init(z)

        # Initial estimate of state
        self.x_hat[0] = z[0]
        # Initial estimate of measure
        self.y_hat[0] = self.H @ self.x_hat[0]

        for t in tqdm(
            range(1, z.shape[0]),
            desc="Kalman Filter Progress",
            leave=True,
            total=z.shape[0] - 1,
        ):
            # ******* Prediction Step *******
            x_hat_minus, P_var_minus, y_hat_minus, V_var_minus = self.__KF_predict(
                self.x_hat[t - 1],
                self.y_hat[t - 1],
                self.P_var[t - 1],
                self.V_var[t - 1],
            )
            if verbose is True:
                print("Estimates: ", "\nx=", x_hat_minus, "\ny=", y_hat_minus)

            # ******* Update Step *******
            (
                x_hat_minus_corrected,
                P_var_minus_corrected,
                y_hat_minus_corrected,
                V_var_minus_corrected,
            ) = self.__KF_correct(
                x=x_hat_minus,
                y=y_hat_minus,
                y_obs=z[t],
                P=P_var_minus,
                V=V_var_minus,
            )
            if verbose is True:
                print(
                    "Corrected: ",
                    "\nx=",
                    x_hat_minus_corrected,
                    "\ny=",
                    y_hat_minus_corrected,
                )
            self.x_hat[t] = x_hat_minus_corrected
            self.P_var[t] = P_var_minus_corrected
            self.y_hat[t] = y_hat_minus_corrected
            self.V_var[t] = V_var_minus_corrected

        return self.x_hat, self.P_var, self.y_hat, self.V_var

    def __KF_predict(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        P: npt.NDArray[np.float64],
        V: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """The Kalman Filter prediction step

        Args:
            x (npt.NDArray[np.float64]): _description_
            y (npt.NDArray[np.float64]): _description_
            P (npt.NDArray[np.float64]): _description_
            V (npt.NDArray[np.float64]): _description_

        Returns:
            Tuple[ npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], ]: _description_
        """
        x = self.F @ x  # + self.B @ self.u
        P = self.F @ P @ self.F.T + self.Q
        y = self.H @ x
        V = self.H @ V @ self.H.T + self.R

        return x, P, y, V

    def __KF_correct(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        y_obs: npt.NDArray[np.float64],
        P: npt.NDArray[np.float64],
        V: npt.NDArray[np.float64],
    ) -> Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """The Kalman Filter correction step

        Args:
            x (npt.NDArray[np.float64]): The predicted state estimate
            y (npt.NDArray[np.float64]): The predicted measure estimate
            y_obs (npt.NDArray[np.float64]): The observed measure
            P (npt.NDArray[np.float64]): The predicted state estimate error variance
            V (npt.NDArray[np.float64]): The predicted measure estimate error variance

        Returns:
            Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64]]: _description_
        """
        K = P @ self.H.T @ np.linalg.pinv(V)
        x = x + K @ (y_obs - y)
        P = (np.eye(P.shape[0]) - K @ self.H) @ P
        return x, P, y, V
