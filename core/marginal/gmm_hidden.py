# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Gaussian mixture distribution marginal inference"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from core.marginal.constants import _2pi, sg0, sqrt2, sqrt2pi, gausshermite


class GMMHidden(Layer):
    """A layer that transform standard sampling (Gauss-Hermite quadrature)
    data points to aligning with the axis of distribution (e.g thru Cholesky
    or Spectral-decomposition) parameterized by mean(u), sigma(o) and correlation(p)
    add distribution variables in this layer, e.g. use mixture of gaussian distribution
    Tensor shape=(C, Node, L, batch_size)

    Args:
        k: number of quadrature points
        c: the number of class of data
        m: grid like MRF height
        n: grid like MRF width
        l: num of mixture components
        dtype: tensor type of distribution variable
    """

    def __init__(self, c, m, l, bs=8, k=17, lr=0.001, dtype=tf.float64):
        super(GMMHidden, self).__init__(name='gaussian_mixture_marginal', dtype=dtype)
        tf.assert_equal(m, 27)
        node = 27 ** 2 + 9 ** 2 + 3 ** 2 + 1
        self.tw = tf.reshape(tf.concat([tf.zeros(729, dtype), tf.ones(90, dtype) * 9, [8.]], 0), [1, 1, -1, 1, 1])
        id1 = tf.reshape(
            tf.tile(tf.reshape(tf.tile(tf.reshape(tf.range(729, 810), [-1, 1]), [1, 3]), [-1, 1, 27]), [1, 3, 1]), [-1])
        id2 = tf.reshape(tf.tile(tf.reshape(tf.tile(tf.reshape(tf.range(810, 819), [-1, 1]), [1, 3]), [-1, 9]), [1, 3]),
                         [-1])
        id3 = tf.ones(9, tf.int32) * 819
        self.id = tf.concat([id1, id2, id3], axis=0)
        self.top0 = tf.zeros([c, 1 + bs, 1, l, 1], dtype)
        self.btm0 = tf.zeros([c, 1 + bs, 729, l, 1], dtype)
        self.s1 = [c, 1 + bs, 9, 3, 9, 3, l, 1]
        self.s2 = [c, 1 + bs, 3, 3, 3, 3, l, 1]
        self.s3 = [c, 1 + bs, -1, l, 1]
        self.xn, self.xi, self.xj, self.wn, self.we = gausshermite(k, dims=[1, 1, 1, 1, -1], dtype=dtype)
        self.x22 = 2 * self.xn ** 2 - 1
        self.xi22 = 2 * self.xi ** 2 - 1
        self.xj22 = 2 * self.xj ** 2 - 1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        _mu_ = tf.tile(tf.reshape(tf.cast(tf.linspace(0., 1., l), dtype), [1, 1, 1, -1, 1]), [c, 1 + bs, node, 1, 1])
        self.mu_initializer = tf.constant_initializer(_mu_.numpy())
        self.mu = self.add_weight(name='mean',
                                  shape=(c, 1 + bs, node, l, 1),
                                  initializer=self.mu_initializer,
                                  # constraint=lambda u: tf.clip_by_value(u, 0, 1),
                                  trainable=True,
                                  dtype=dtype)
        self.sigma_initializer = tf.constant_initializer(1.)
        self.sigma = self.add_weight(name='standard_deviation',
                                     shape=(c, 1 + bs, node, l, 1),
                                     initializer=self.sigma_initializer,
                                     constraint=lambda o: tf.clip_by_value(o, 0.01, 50.),
                                     trainable=True,
                                     dtype=dtype)
        # mixture weights alpha can be represented as a softmax of unconstrained variables w
        self.w_initializer = tf.constant_initializer(0.)
        self.w = self.add_weight(name='mixture_component_weight',
                                 shape=(c, 1 + bs, 1, l, 1),
                                 initializer=self.w_initializer,
                                 constraint=lambda x: tf.clip_by_value(x, -300, 300),
                                 trainable=True,
                                 dtype=dtype)
        self.built = True

    def init_marginal(self):
        for k, initializer in self.__dict__.items():
            if "initializer" in k:
                var = self.__getattribute__(k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
        # reset optimizer, for adam 1st momentum(m) and 2nd momentum(v)
        if self.optimizer.get_weights():
            self.optimizer.set_weights([tf.constant(0).numpy()] +
                                       [tf.zeros_like(v).numpy() for v in self.trainable_variables] * 2)

    def conditioned(self, x0):
        self.mu[:, 1:, :729, ...].assign(x0)
        self.sigma[:, 1:, :729, ...].assign(tf.zeros_like(x0))

    def call(self, inputs, full_out=True):
        """transform quadrature points to align with current distribution
        the inputs is a list of tensors of the sampling points:
        inputs[0] is the xn, gauss-hermite quadrature points vector for node sampling
        inputs[1] is the stack of (xi, xj), from the mesh-grid coordinates of points vector
        """
        xn = self.xn * self.sigma * sqrt2 + self.mu
        u1 = self.mu[:, :, :-1, ...]
        o1 = self.sigma[:, :, :-1, ...]
        u2 = tf.gather(self.mu, self.id, axis=2)
        o2 = tf.gather(self.sigma, self.id, axis=2)
        x1 = self.xi * o1 * sqrt2 + u1
        x2 = self.xj * o2 * sqrt2 + u2
        xe = tf.stack([x1, x2], -1)
        # ------------------------------log pdf -> log of sum of each mixture component-------------------------------
        alpha = tf.nn.softmax(self.w, axis=3)
        alf = tf.expand_dims(alpha, 3)
        o = tf.expand_dims(self.sigma, 3)
        # lpn = tf.math.divide_no_nan(tf.exp(-((tf.expand_dims(xn, 4) -
        #                                       tf.expand_dims(self.mu, 3)) / o) ** 2 / 2), o * sqrt2pi)
        lpn = tf.exp(-((tf.expand_dims(xn, 4) - tf.expand_dims(self.mu, 3)) / o) ** 2 / 2) / (o * sqrt2pi)
        lpn = tf.math.log(tf.reduce_sum(lpn * alf, 4) + 1e-307)
        lpn = tf.concat(  # intercept NaN -> 0
            [tf.concat([lpn[:, :1, :729, ...], tf.zeros_like(lpn)[:, 1:, :729, ...]], 1), lpn[:, :, 729:, ...]], 2)
        o1_ = tf.expand_dims(o1, 3)
        o2_ = tf.expand_dims(o2, 3)
        b1 = tf.exp(-((tf.expand_dims(x1, 4) - tf.expand_dims(u1, 3)) / o1_) ** 2 / 2) / (o1_ * sqrt2pi)
        b1 = tf.concat(  # intercept NaN -> 1
            [tf.concat([b1[:, :1, :729, ...], tf.ones_like(b1)[:, 1:, :729, ...]], 1), b1[:, :, 729:, ...]], 2)
        b2 = tf.exp(-((tf.expand_dims(x2, 4) - tf.expand_dims(u2, 3)) / o2_) ** 2 / 2) / (o2_ * sqrt2pi)
        # lpe = tf.math.divide_no_nan(tf.exp(-((tf.expand_dims(x1, 4) - tf.expand_dims(u1, 3)) / o1_) ** 2 / 2
        #                                    - ((tf.expand_dims(x2, 4) - tf.expand_dims(u2, 3)) / o2_) ** 2 / 2),
        #                             o1_ * o2_ * _2pi)
        lpe = tf.math.log(tf.reduce_sum(b1 * b2 * alf, 4) + 1e-307)
        return (xn, xe, lpn, lpe, alpha, o1, o2) if full_out else (xn, xe, lpn, lpe, alpha)

    @tf.function
    def predict(self, x):
        alpha = tf.nn.softmax(self.w[:, :1, ...], axis=3)
        u = self.mu[:, :1, :729, ...]
        o = self.sigma[:, :1, :729, ...]
        x = -((x - u) / o) ** 2 / 2
        prob = tf.reduce_sum(tf.reduce_prod(tf.exp(x) / o, 2, keepdims=True) * alpha, 3)

        # could solve product -> 0 problem
        log_con = tf.reduce_sum(x - tf.math.log(o), axis=2, keepdims=True) + tf.math.log(alpha)
        max_con = tf.reduce_max(log_con, axis=3, keepdims=True)
        log_prob = max_con + tf.math.log(tf.reduce_sum(tf.exp(log_con - max_con), 3, True) + 1e-300)
        return tf.squeeze(prob), tf.squeeze(log_prob)

    @tf.function
    def bethe_free_energy(self, potential):
        """calculate the bethe free energy which contains expectation term and entropy term
        Args:
            potential: a callable function, could be a neural net layer
        Returns:
            tensor representing the bethe free energy to be minimized, shape = (C,)
        """
        xn, xe, lpn, lpe, alpha = self(None, full_out=False)
        fn, fe = potential((xn, xe))
        bfe = -(tf.reduce_sum((fn + self.tw * lpn) * self.wn * alpha, [2, 3, 4]) +
                tf.reduce_sum((fe - lpe) * self.we * alpha, [2, 3, 4]))
        return bfe

    @tf.function
    def potential_expectation(self, potential):
        xn = self.xn * self.sigma * sqrt2 + self.mu
        u1 = self.mu[:, :, :-1, ...]
        o1 = self.sigma[:, :, :-1, ...]
        u2 = tf.gather(self.mu, self.id, axis=2)
        o2 = tf.gather(self.sigma, self.id, axis=2)
        x1 = self.xi * o1 * sqrt2 + u1
        x2 = self.xj * o2 * sqrt2 + u2
        xe = tf.stack([x1, x2], -1)
        alpha = tf.nn.softmax(self.w, axis=3)

        fn, fe = potential((xn, xe))
        return tf.reduce_sum(alpha * self.wn * fn, [2, 3, 4]) + tf.reduce_sum(alpha * self.we * fe, [2, 3, 4])

    @tf.function
    def infer(self, potential, iterations, cf=True, x0=None):
        """infer the marginal distribution out by minimizing the energy of mrf, e.g bethe free energy
        Args:
            potential: potential functions of cliques, probably a neural net layer which is callable
            iterations: an integer that specify how many iterations to perform the inference
            cf: use close form method to get gradient if true otherwise tf auto-gradient
            x0: observed data for non hidden nodes, shape=[c, bs, node, 1, 1]
        Returns:
            stores the final energy of current state, can be used as logZ approximation
        """
        energy = tf.zeros(shape=[x0.shape[0], 1 + x0.shape[1]], dtype=self.dtype)
        x0 = tf.tile(x0, [1, 1, 1, self.mu.shape[-2], 1])
        for i in tf.range(iterations):
            self.conditioned(x0)
            verbose = tf.equal(tf.math.mod(i, 50), 0) or tf.equal(i + 1, iterations)
            if cf:
                grd, energy = self.gradient_cf(potential, verbose)
            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(self.trainable_variables)
                    energy = self.bethe_free_energy(potential)
                grd = tape.gradient(energy, self.trainable_variables)

            self.optimizer.apply_gradients(zip(grd, self.trainable_variables))
            # reasonable clip for the distribution of bottom layer
            self.mu[:, 0, :729, ...].assign(tf.clip_by_value(self.mu[:, 0, :729, ...], 0., 1.))
            self.sigma[:, 0, :729, ...].assign(tf.clip_by_value(self.sigma[:, 0, :729, ...], 0.02, 1.))

            if verbose:
                tf.print(tf.strings.format('iter: {} dmu = {}, dsigma = {}, dw={}, Energy = {}', (
                    i, tf.reduce_mean(tf.abs(grd[0])), tf.reduce_mean(tf.abs(grd[1])), tf.reduce_mean(tf.abs(grd[2])),
                    tf.reduce_mean(energy))))
        return energy

    @tf.function
    def gradient_cf(self, potential, get_energy=True):
        """compute the close form gradient of bfe"""
        xn, xe, lpn, lpe, alpha, o1, o2 = self(None)
        fn_, fe_ = potential((xn, xe))
        fn_ = (fn_ + self.tw * lpn) * self.wn
        fe_ = (fe_ - lpe) * self.we
        fn = fn_ * alpha
        fe = fe_ * alpha
        dmu = tf.math.divide_no_nan(tf.reduce_sum(fn * self.xn, axis=-1, keepdims=True), self.sigma)
        dsg = tf.math.divide_no_nan(tf.reduce_sum(fn * self.x22, axis=-1, keepdims=True), self.sigma)
        dmu1 = tf.math.divide_no_nan(tf.reduce_sum(fe * self.xi, -1, keepdims=True), o1)
        dmu2 = tf.reduce_sum(fe * self.xj, -1, keepdims=True) / o2
        dsg1 = tf.math.divide_no_nan(tf.reduce_sum(fe * self.xi22, -1, keepdims=True), o1)
        dsg2 = tf.reduce_sum(fe * self.xj22, -1, keepdims=True) / o2

        dmu += (tf.concat([dmu1, self.top0], 2) + tf.concat(
            [self.btm0, tf.reshape(tf.reduce_sum(tf.reshape(dmu2[:, :, :729, ...], self.s1), [3, 5]), self.s3),
             tf.reshape(tf.reduce_sum(tf.reshape(dmu2[:, :, 729:810, ...], self.s2), [3, 5]), self.s3),
             tf.reduce_sum(dmu2[:, :, 810:, ...], 2, True)], 2))

        dsg += (tf.concat([dsg1, self.top0], 2) + tf.concat(
            [self.btm0, tf.reshape(tf.reduce_sum(tf.reshape(dsg2[:, :, :729, ...], self.s1), [3, 5]), self.s3),
             tf.reshape(tf.reduce_sum(tf.reshape(dsg2[:, :, 729:810, ...], self.s2), [3, 5]), self.s3),
             tf.reduce_sum(dsg2[:, :, 810:, ...], 2, True)], 2))

        dalpha = (tf.reduce_sum(fn_, [2, 4], keepdims=True) + tf.reduce_sum(fe_, [2, 4], keepdims=True))
        dw = alpha * (dalpha - tf.reduce_sum(dalpha * alpha, 3, keepdims=True))
        energy = tf.zeros(fn.shape[:2], tf.float64) if not get_energy else \
            -(tf.reduce_sum(fn, [2, 3, 4]) + tf.reduce_sum(fe, [2, 3, 4]))
        return (-dmu * sqrt2, -dsg, -dw), energy


if __name__ == '__main__':
    from core.potential.dnn import NeuralNetPotential
    import timeit

    gmm = GMMHidden(c=10, m=27, l=1, k=11)
    pot = NeuralNetPotential(node_units=(4, 5, 5, 4), edge_units=(5, 7, 7, 5))
    print(timeit.timeit(lambda: gmm.infer(pot, 50), number=1))
