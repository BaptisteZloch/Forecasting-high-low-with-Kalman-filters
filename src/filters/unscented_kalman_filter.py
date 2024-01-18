import numpy as np
import numpy.typing as npt


class UKF:
    def __init__(self, dim_x, dim_z, Q, R, kappa=0.0):
        """
        UKF class constructor
        inputs:
            dim_x : state vector x dimension
            dim_z : measurement vector z dimension

        - step 1: setting dimensions
        - step 2: setting number of sigma points to be generated
        - step 3: setting scaling parameters
        - step 4: calculate scaling coefficient for selecting sigma points
        - step 5: calculate weights
        """

        # setting dimensions
        self.dim_x = dim_x  # state dimension
        self.dim_z = dim_z  # measurement dimension
        self.dim_v = np.shape(Q)[0]
        self.dim_n = np.shape(R)[0]
        self.dim_a = (
            self.dim_x + self.dim_v + self.dim_n
        )  # assuming noise dimension is same as x dimension

        # setting number of sigma points to be generated
        self.n_sigma = (2 * self.dim_a) + 1

        # setting scaling parameters
        self.kappa = 3 - self.dim_a  # kappa
        self.alpha = 0.001
        self.beta = 2.0

        alpha_2 = self.alpha**2
        self.lambda_ = alpha_2 * (self.dim_a + self.kappa) - self.dim_a

        # setting scale coefficient for selecting sigma points
        # self.sigma_scale = np.sqrt(self.dim_a + self.lambda_)
        self.sigma_scale = np.sqrt(self.dim_a + self.kappa)

        # calculate unscented weights
        # self.W0m = self.W0c = self.lambda_ / (self.dim_a + self.lambda_)
        # self.W0c = self.W0c + (1.0 - alpha_2 + self.beta)
        # self.Wi = 0.5 / (self.dim_a + self.lambda_)

        self.W0 = self.kappa / (self.dim_a + self.kappa)
        self.Wi = 0.5 / (self.dim_a + self.kappa)

        # initializing augmented state x_a and augmented covariance P_a
        self.x_a = np.zeros((self.dim_a,))
        self.P_a = np.zeros((self.dim_a, self.dim_a))

        self.idx1, self.idx2 = self.dim_x, self.dim_x + self.dim_v

        self.P_a[self.idx1 : self.idx2, self.idx1 : self.idx2] = Q
        self.P_a[self.idx2 :, self.idx2 :] = R

        print(f"P_a = \n{self.P_a}\n")

    def predict(self, f, x, P):
        self.x_a[: self.dim_x] = x
        self.P_a[: self.dim_x, : self.dim_x] = P

        xa_sigmas = self.sigma_points(self.x_a, self.P_a)

        xx_sigmas = xa_sigmas[: self.dim_x, :]
        xv_sigmas = xa_sigmas[self.idx1 : self.idx2, :]

        y_sigmas = np.zeros((self.dim_x, self.n_sigma))
        for i in range(self.n_sigma):
            y_sigmas[:, i] = f(xx_sigmas[:, i], xv_sigmas[:, i])

        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)

        self.x_a[: self.dim_x] = y
        self.P_a[: self.dim_x, : self.dim_x] = Pyy

        return y, Pyy, xx_sigmas

    def correct(self, h, x, P, z):
        self.x_a[: self.dim_x] = x
        self.P_a[: self.dim_x, : self.dim_x] = P

        xa_sigmas = self.sigma_points(self.x_a, self.P_a)

        xx_sigmas = xa_sigmas[: self.dim_x, :]
        xn_sigmas = xa_sigmas[self.idx2 :, :]

        y_sigmas = np.zeros((self.dim_z, self.n_sigma))
        for i in range(self.n_sigma):
            y_sigmas[:, i] = h(xx_sigmas[:, i], xn_sigmas[:, i])

        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)

        Pxy = self.calculate_cross_correlation(x, xx_sigmas, y, y_sigmas)

        K = Pxy @ np.linalg.pinv(Pyy)

        x = x + (K @ (z - y))
        P = P - (K @ Pyy @ K.T)

        return x, P, xx_sigmas

    def sigma_points(self, x, P):
        """
        generating sigma points matrix x_sigma given mean 'x' and covariance 'P'
        """

        nx = np.shape(x)[0]

        x_sigma = np.zeros((nx, self.n_sigma))
        x_sigma[:, 0] = x

        S = np.linalg.cholesky(P)

        for i in range(nx):
            x_sigma[:, i + 1] = x + (self.sigma_scale * S[:, i])
            x_sigma[:, i + nx + 1] = x - (self.sigma_scale * S[:, i])

        return x_sigma

    def calculate_mean_and_covariance(self, y_sigmas):
        ydim = np.shape(y_sigmas)[0]

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

    def calculate_cross_correlation(self, x, x_sigmas, y, y_sigmas):
        xdim = np.shape(x)[0]
        ydim = np.shape(y)[0]

        n_sigmas = np.shape(x_sigmas)[1]

        dx = (x_sigmas[:, 0] - x).reshape([-1, 1])
        dy = (y_sigmas[:, 0] - y).reshape([-1, 1])
        Pxy = self.W0 * (dx @ dy.T)
        for i in range(1, n_sigmas):
            dx = (x_sigmas[:, i] - x).reshape([-1, 1])
            dy = (y_sigmas[:, i] - y).reshape([-1, 1])
            Pxy += self.Wi * (dx @ dy.T)

        return Pxy
