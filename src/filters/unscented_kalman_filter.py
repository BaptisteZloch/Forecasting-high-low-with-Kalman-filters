from typing import Callable, Optional, Tuple
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import scipy


class UnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, Q, R):
        """UKF class constructor
        - step 1: setting dimensions
        - step 2: setting number of sigma points to be generated
        - step 3: setting scaling parameters
        - step 4: calculate scaling coefficient for selecting sigma points

        Args:
            dim_x (int): state vector x dimension
            dim_z (int): measurement vector z dimension
            Q (npt.NDArray[np.float64]): The process noise covariance matrix (shape: (n_features, n_features))
            R (npt.NDArray[np.float64]): The measurement noise covariance matrix (shape: (n_features, n_features))
        """

        # setting dimensions
        self.dim_x = dim_x  # state dimension
        self.dim_z = dim_z  # measurement dimension
        self.dim_v = Q.shape[0]
        self.dim_u = Q.shape[0]
        self.dim_n = R.shape[0]
        self.dim_a = (
            self.dim_x + self.dim_v + self.dim_n
        )  # assuming noise dimension is same as x dimension

        # setting number of sigma points to be generated
        self.n_sigma = (2 * self.dim_a) + 1

        # setting scaling parameters
        self.kappa = 1  # 3 - self.dim_a
        self.alpha = 0.001
        self.beta = 2.0

        self.lambda_ = (self.alpha**2) * (self.dim_a + self.kappa) - self.dim_a

        # setting scale coefficient for selecting sigma points
        self.sigma_scale = np.sqrt(self.dim_a + self.kappa)

        # calculate unscented weights
        self.W0 = self.lambda_ / (self.dim_a + self.lambda_)
        # self.W0 = self.kappa / (self.dim_a + self.kappa)
        # self.Wi = 0.5 / (self.dim_a + self.kappa)
        self.Wi = 0.5 / (self.dim_a + self.lambda_)

        # initializing augmented state x_a and augmented covariance P_a
        self.x_a = np.zeros((self.dim_a,))
        self.P_a = np.zeros((self.dim_a, self.dim_a))

        self.idx1, self.idx2 = self.dim_x, self.dim_x + self.dim_v

        self.P_a[self.idx1 : self.idx2, self.idx1 : self.idx2] = Q
        self.P_a[self.idx2 :, self.idx2 :] = R

    def fit_predict(
        self,
        state_function: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
            npt.NDArray[np.float64],
        ],
        observation_function: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
            npt.NDArray[np.float64],
        ],
        z: npt.NDArray[np.float64],
        u: Optional[npt.NDArray[np.float64]] = None,
        x0: Optional[npt.NDArray[np.float64]] = None,
        verbose: bool = False,
        keep_state_estimates: bool = False,
    ) -> Tuple[npt.NDArray[np.float64], ...]:
        """Fit the UKF on the observations dataset and return the filtered state vector and the filtered state covariance matrix

        Args:
        -----
            state_function (Callable[ [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64], ]): The state function that take as input 3 vectors: The previous state vector, the control vector, the process noise and that returns a vector (shape: (n_features,))
            observation_function (Callable[ [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64], ]): The observation function that take as input 3 vectors: The current state vector, the control vector, the measurement noise and that returns a vector (shape: (n_features,))
            z (npt.NDArray[np.float64]): The observations dataset to fit the filter on (shape: (n_samples, n_features))
            u (Optional[npt.NDArray[np.float64]], optional): The control vector dataset same shape as z By default None is provided. Defaults to None.

        Returns:
        ----
            Tuple[npt.NDArray[np.float64],...]: The filtered state vector a each step (shape: (n_samples, n_features)), the filtered state covariance matrix (shape: (n_samples, n_features, n_features))
        """
        self.verbose = verbose
        if u is None:
            u = np.zeros(shape=z.shape)

        X_hat = np.zeros((z.shape[0], self.dim_x))
        X_hat[0] = x0
        P_var = np.zeros((z.shape[0], self.dim_x, self.dim_x))
        P_var[0] = np.eye(P_var[0].shape[0])
        if keep_state_estimates:
            X_hat_estim = np.zeros(X_hat.shape)
            X_hat_estim[0] = X_hat[0]
        for i in tqdm(
            range(1, z.shape[0]), desc="UKF", leave=True, total=z.shape[0] - 1
        ):
            x1, P1, _ = self.__UKF_predict(
                state_function,
                X_hat[i - 1],
                P_var[i - 1],
                u[i - 1],
            )

            if keep_state_estimates is True:
                X_hat_estim[i] = x1
            X_hat[i], P_var[i], _ = self.__UKF_correct(
                observation_function,
                x1,
                P1,
                z[i],
                u[i - 1],
            )
        if keep_state_estimates:
            return X_hat, P_var, X_hat_estim
        return X_hat, P_var

    def __UKF_predict(
        self,
        state_function: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
            npt.NDArray[np.float64],
        ],
        x: npt.NDArray[np.float64],
        P: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], ...]:
        """UKF prediction step

        Args:
            state_function (Callable[ [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64], ]): _description_
            x (npt.NDArray[np.float64]): The current state vector (shape: (n_features,))
            P (npt.NDArray[np.float64]): The current state covariance matrix (shape: (n_features, n_features))
            u (npt.NDArray[np.float64]): The control vector (shape: (n_features,))

        Returns:
            Tuple[npt.NDArray[np.float64], ...]: The predicted state vector (shape: (n_features,)), the predicted state covariance matrix (shape: (n_features, n_features)), the sigma points matrix (shape: (n_features, n_sigma))
        """
        self.x_a[: self.dim_x] = x
        self.P_a[: self.dim_x, : self.dim_x] = P

        xa_sigmas = self.__sigma_points(self.x_a, self.P_a)
        xx_sigmas = xa_sigmas[: self.dim_x, :]
        xv_sigmas = xa_sigmas[self.idx1 : self.idx2, :]

        y_sigmas = np.zeros((self.dim_x, self.n_sigma))
        for i in range(self.n_sigma):
            y_sigmas[:, i] = state_function(xx_sigmas[:, i], u, xv_sigmas[:, i])

        y, Pyy = self.__calculate_mean_and_covariance(y_sigmas)

        self.x_a[: self.dim_x] = y
        self.P_a[: self.dim_x, : self.dim_x] = Pyy

        return y, Pyy, xx_sigmas

    def __UKF_correct(
        self,
        observation_function: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]],
            npt.NDArray[np.float64],
        ],
        x: npt.NDArray[np.float64],
        P: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], ...]:
        """The correction step of the UKF

        Args:
        ----
            observation_function (Callable[ [npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64], ]): The observation function that take as input 3 vectors: The current state vector, the control vector, the measurement noise and that returns a vector (shape: (n_features,))
            x (npt.NDArray[np.float64]): The current state vector (shape: (n_features,))
            P (npt.NDArray[np.float64]): The current state covariance matrix (shape: (n_features, n_features))
            z (npt.NDArray[np.float64]): The current observation vector (shape: (n_features,))
            u (npt.NDArray[np.float64]): The control vector (shape: (n_features,)).

        Returns:
        ----
            Tuple[npt.NDArray[np.float64], ...]: _description_
        """
        self.x_a[: self.dim_x] = x
        self.P_a[: self.dim_x, : self.dim_x] = P

        xa_sigmas = self.__sigma_points(self.x_a, self.P_a)

        xx_sigmas = xa_sigmas[: self.dim_x, :]
        xn_sigmas = xa_sigmas[self.idx2 :, :]

        y_sigmas = np.zeros((self.dim_z, self.n_sigma))
        for i in range(self.n_sigma):
            y_sigmas[:, i] = observation_function(xx_sigmas[:, i], u, xn_sigmas[:, i])
        if self.verbose:
            print(f"y_sigmas={y_sigmas}")
        y, Pyy = self.__calculate_mean_and_covariance(y_sigmas)

        Pxy = self.__calculate_cross_correlation(x, xx_sigmas, y, y_sigmas)

        K = Pxy @ np.linalg.pinv(Pyy)
        if self.verbose:
            print("x=", x, "\nK=", K, "\nz=", z, "\ny=", y)
        x = x + (K @ (z - y))
        P = P - (K @ Pyy @ K.T)

        return x, P, xx_sigmas

    def __sigma_points(
        self, x: npt.NDArray[np.float64], P: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """generating sigma points matrix x_sigma given mean 'x' and covariance 'P'

        Args:
        ----
            x (npt.NDArray[np.float64]): The current state vector (shape: (n_features,))
            P (npt.NDArray[np.float64]): The current state covariance matrix (shape: (n_features, n_features))

        Returns:
        ----
            npt.NDArray[np.float64]: The sigma points matrix
        """
        nx = x.shape[0]

        x_sigma = np.zeros((nx, self.n_sigma))
        x_sigma[:, 0] = x
        if self.verbose:
            print(f"P={P}")

        try:
            S = np.linalg.cholesky(P)
        except:
            _, S, _ = scipy.linalg.lu(P)
        for i in range(nx):
            x_sigma[:, i + 1] = x + (self.sigma_scale * S[:, i])
            x_sigma[:, i + nx + 1] = x - (self.sigma_scale * S[:, i])

        return x_sigma

    def __calculate_mean_and_covariance(
        self, y_sigmas: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate mean and covariance of sigma points

        Args:
        ----
            y_sigmas (npt.NDArray[np.float64]): The sigma points matrix

        Returns:
        ----
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: The mean vector and the covariance matrix
        """
        # mean calculation
        y = self.W0 * y_sigmas[:, 0]
        for i in range(1, self.n_sigma):
            y += self.Wi * y_sigmas[:, i]

        # covariance calculation
        d = (y_sigmas[:, 0] - y).reshape([-1, 1])
        Pyy = self.W0 * (d @ d.T)
        for i in range(1, self.n_sigma):
            d = (y_sigmas[:, i] - y).reshape([-1, 1])
            Pyy += self.Wi * (d @ d.T)

        return y, Pyy

    def __calculate_cross_correlation(
        self,
        x: npt.NDArray[np.float64],
        x_sigmas: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        y_sigmas: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate cross correlation between x and y

        Args:
            x (npt.NDArray[np.float64]): The current state vector (shape: (n_features,))
            x_sigmas (npt.NDArray[np.float64]): The sigma points matrix
            y (npt.NDArray[np.float64]): The mean vector
            y_sigmas (npt.NDArray[np.float64]): The sigma points matrix

        Returns:
            npt.NDArray[np.float64]: The cross correlation matrix
        """
        n_sigmas = x_sigmas.shape[1]

        dx = (x_sigmas[:, 0] - x).reshape([-1, 1])
        dy = (y_sigmas[:, 0] - y).reshape([-1, 1])
        Pxy = self.W0 * (dx @ dy.T)
        for i in range(1, n_sigmas):
            dx = (x_sigmas[:, i] - x).reshape([-1, 1])
            dy = (y_sigmas[:, i] - y).reshape([-1, 1])
            Pxy += self.Wi * (dx @ dy.T)

        return Pxy
