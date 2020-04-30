"""file define useful utility functions"""
import math
import tensorflow as tf
from numpy.polynomial.hermite import hermgauss

pi = math.pi

_2pi = 2 * pi

sg0 = 1 / math.sqrt(_2pi * math.e)

sqrtpi = math.sqrt(pi)

sqrt2 = math.sqrt(2)

sqrt2pi = math.sqrt(_2pi)

# normal pdf constant
ncn = -math.log(_2pi) / 2

# normal distribution entropy constant
ndc = (1 + math.log(_2pi)) / 2


def gausshermite(k, dims=None, dtype=tf.float64):
    """Gauss-Hermite function to get sampling points and weights

        Argument:
        k: number of quadrature points.

        Return:
        x: the vector contains quadrature points
        w: the weights of corresponding points
    """
    if dims is None:
        dims = [1, 1, 1, 1, 1, -1]  # Tensor shape = (C, M, N, cliques/unit, L, batch_size)
    x, w = hermgauss(k)
    xn = tf.cast(tf.reshape(x, dims), dtype)
    wn = tf.cast(tf.reshape(w / sqrtpi, dims), dtype)
    xi, xj = tf.meshgrid(x, x)
    wi, wj = tf.meshgrid(w, w)
    xi = tf.cast(tf.reshape(xi, dims), dtype)
    xj = tf.cast(tf.reshape(xj, dims), dtype)
    we = tf.cast(tf.reshape(wi * wj / pi, dims), dtype)
    return xn, xi, xj, wn, we
