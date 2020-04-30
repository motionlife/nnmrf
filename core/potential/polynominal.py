# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""module define polynomial potential function layer"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer


class PolynomialPotential(Layer):
    """served as baseline potential function for testing and comparision purpose"""

    def __init__(self, order=2, initializer='glorot_normal', name='polynomial', dtype=tf.float64):
        super(PolynomialPotential, self).__init__(name=name, dtype=dtype)
        tf.assert_equal(order, 2)  # higher order Polynomial Potential not implemented yet
        self.order = order
        self.initializer = initializer

    def build(self, input_shape):
        if input_shape[1].rank == 6:  # hidden variable structure
            ls0, ls1 = input_shape[0].as_list(), input_shape[1].as_list()
            ls0[1], ls1[1] = 1, 1
            input_shape[0], input_shape[1] = tf.TensorShape(ls0), tf.TensorShape(ls1)

        shp0 = input_shape[0][:-2].concatenate([1, 1, self.order])
        shp1 = input_shape[1][:-3].concatenate([1, 1])

        # build custom initializer
        n_pots0, n_pots1 = tf.reduce_prod(shp0[:-1]).numpy(), tf.reduce_prod(shp1[:-1]).numpy()
        scale0, scale1, mode = (2 * n_pots0, 2 * n_pots1, "fan_in") if "he" in self.initializer else (
            n_pots0, n_pots1, "fan_avg")
        distribution = "uniform" if "uniform" in self.initializer else "truncated_normal"
        self.initializer0, self.initializer1 = tf.keras.initializers.VarianceScaling(scale0, mode, distribution), \
                                               tf.keras.initializers.VarianceScaling(scale1, mode, distribution)
        self.nth = self.add_weight(name='node_theta',
                                   shape=shp0,
                                   initializer=self.initializer0,
                                   regularizer=None,
                                   trainable=True)
        self.eth = self.add_weight(name='edge_theta',
                                   shape=shp1,
                                   initializer=self.initializer1,
                                   regularizer=None,
                                   trainable=True)
        self.built = True

    def call(self, inputs):
        fn = inputs[0]
        fe = inputs[1]
        fn = (self.nth[..., 0] + fn * self.nth[..., 1]) * fn  # - fn ** 4
        fe = tf.reduce_prod(fe, -1)
        fe = fe * self.eth  # - fe ** 2
        # fn = (self.nth[..., 0] + fn * self.nth[..., 1] + fn ** 2 * self.nth[..., 2] if self.order == 3 else 0.) * fn
        # fe = tf.reduce_prod(fe, -1, keepdims=True) * (tf.expand_dims(
        #     self.eth[..., 0] + (
        #             self.eth[..., 1] * fe[..., 0] + self.eth[..., 2] * fe[..., 1]) if self.order == 3 else 0., -1))
        return fn, fe
