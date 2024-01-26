from typing import Optional, Tuple
import tensorflow as tf
import numpy as np
import numpy.typing as npt
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant


class InitCentersRandom(Initializer):
    """Initializer for initialization of centers of RBF network
        as random samples from the given data set.

    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X: npt.NDArray[np.float64]):
        self.X = X
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]  # check dimension

        # np.random.randint returns ints from [low, high) !
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(tf.keras.layers.Layer):
    """Layer of Gaussian RBF units.

    # Example

    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```

    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas

    """

    def __init__(
        self,
        output_dim: int,
        initializer: Optional[Initializer] = None,
        betas: float = 1.0,
        **kwargs
    ):
        self.output_dim = output_dim

        # betas is either initializer object or float
        if isinstance(betas, Initializer):
            self.betas_initializer = betas
        else:
            self.betas_initializer = Constant(value=betas)

        self.initializer = (
            initializer if initializer is not None else RandomUniform(0.0, 1.0)
        )

        super().__init__(**kwargs)

    def build(self, input_shape: Tuple[int, int]):
        self.centers = self.add_weight(
            name="centers",
            shape=(self.output_dim, input_shape[1]),
            initializer=self.initializer,
            trainable=True,
        )
        self.betas = self.add_weight(
            name="betas",
            shape=(self.output_dim,),
            initializer=self.betas_initializer,
            # initializer='ones',
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x):
        C = tf.expand_dims(self.centers, -1)  # inserts a dimension of 1
        H = tf.transpose(C - tf.transpose(x))  # matrix of differences
        return tf.exp(-self.betas * tf.math.reduce_sum(H**2, axis=1))

    def compute_output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {"output_dim": self.output_dim}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
