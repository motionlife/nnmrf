# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Gaussian mixture distribution marginal inference for complete graphical model"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from core.marginal.constants import pi, sqrt2, sqrt2pi, gausshermite, ndc, ncn


class GMMComplete(Layer):
    """A layer that transform standard sampling (Gauss-Hermite quadrature)
    data points to aligning with the axis of distribution (e.g thru Cholesky
    or Spectral-decomposition) parameterized by mean(u), sigma(o) and correlation(p)
    add distribution variables in this layer, e.g. use mixture of gaussian distribution

    Args:
        k: number of quadrature points
        c: the number of class of data
        n: node number
        l: num of mixture components
        dtype: tensor type of distribution variable
    """

    def __init__(self, c, n, l, k=17, lr=0.001, dtype=tf.float64):
        super(GMMComplete, self).__init__(name='gaussian_mixture_marginal', dtype=dtype)
        self.N = n
        self.tw = n - 2
        self.id1 = tf.concat([tf.tile([i], [n - 1 - i]) for i in range(n)], 0)
        self.id2 = tf.concat([1 + tf.range(i, n - 1) for i in range(n)], 0)
        self.zeros = tf.zeros([1, c, l, 1], dtype=dtype)
        self.xn, self.xi, self.xj, self.wn, self.we = gausshermite(k, dims=[1, 1, 1, -1], dtype=dtype)
        self.x22 = 2 * self.xn ** 2 - 1
        self.xij = self.xi * self.xj
        self.xi2axj2 = self.xi ** 2 + self.xj ** 2 - 1
        self.xi2mxj2 = self.xi ** 2 - self.xj ** 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        _mu_ = tf.tile(tf.reshape(tf.cast(tf.linspace(-2., 2., l), dtype), [1, 1, -1, 1]), [c, n, 1, 1])
        # _mu_ = tf.reshape(tf.constant([[-1, 1], [1, -1]], dtype=dtype), [c, n, l, 1])
        self.mu_initializer = tf.constant_initializer(_mu_.numpy())
        self.mu = self.add_weight(name='mean',
                                  shape=(c, n, l, 1),
                                  initializer=self.mu_initializer,
                                  constraint=lambda u: tf.clip_by_value(u, -10, 10),
                                  trainable=True,
                                  dtype=dtype)
        # _sigma_ = tf.reshape(tf.constant([[1, 0.5], [0.5, 0.8]], dtype=dtype), [c, n, l, 1])
        self.sigma_initializer = tf.constant_initializer(0.7)
        self.sigma = self.add_weight(name='standard_deviation',
                                     shape=(c, n, l, 1),
                                     initializer=self.sigma_initializer,
                                     constraint=lambda o: tf.clip_by_value(o, 0.01, 10),
                                     trainable=True,
                                     dtype=dtype)
        # _rou_ = tf.reshape(tf.constant([-0.54, 0.38], dtype=dtype), [1, 1, 2, 1])
        self.rou_initializer = tf.constant_initializer(0.)
        self.rou = self.add_weight(name='correlation_coefficient',
                                   shape=(c, n * (n - 1) // 2, l, 1),
                                   initializer=self.rou_initializer,
                                   constraint=lambda p: tf.clip_by_value(p, -0.998, 0.998),
                                   trainable=True,
                                   dtype=dtype)
        # mixture weights alpha can be represented as a softmax of unconstrained variables w
        # _w_ = tf.reshape(tf.constant([0.1527025, 1], dtype=dtype), (c, 1, l, 1))
        self.w_initializer = tf.constant_initializer(0.)
        self.w = self.add_weight(name='mixture_component_weight',
                                 shape=(c, 1, l, 1),
                                 initializer=self.w_initializer,
                                 constraint=lambda x: tf.clip_by_value(x, -300, 300),
                                 trainable=True,
                                 dtype=dtype)
        self.wsp = (c, 1, -1, l, 1)
        self.built = True

    def init_marginal(self):
        for k, initializer in self.__dict__.items():
            if "initializer" in k:
                var = self.__getattribute__(k.replace("_initializer", ""))
                var.assign(initializer(var.shape, var.dtype))
        # reset optimizer, for adam 1st momentum(m) and 2nd momentum(v)
        if self.optimizer.weights:
            self.optimizer.set_weights([tf.constant(0).numpy()] +
                                       [tf.zeros_like(v).numpy() for v in self.trainable_variables] * 2)

    def call(self, inputs, entropy=True, full_out=True):
        """transform quadrature points to align with current distribution
        the inputs is a list of tensors of the sampling points:
        inputs[0] is the xn, gauss-hermite quadrature points vector for node sampling
        inputs[1] is the stack of (xi, xj), from the mesh-grid coordinates of points vector
        """
        if inputs is None: inputs = self.trainable_variables
        xn = self.xn * inputs[1] * sqrt2 + inputs[0]
        s = tf.sqrt(1 + inputs[2]) / 2
        t = tf.sqrt(1 - inputs[2]) / 2
        a = s + t
        b = s - t
        z1 = a * self.xi + b * self.xj
        z2 = b * self.xi + a * self.xj
        u1 = tf.gather(inputs[0], self.id1, axis=1)
        u2 = tf.gather(inputs[0], self.id2, axis=1)
        o1 = tf.gather(inputs[1], self.id1, axis=1)
        o2 = tf.gather(inputs[1], self.id2, axis=1)
        x1 = z1 * o1 * sqrt2 + u1
        x2 = z2 * o2 * sqrt2 + u2
        xe = tf.stack([x1, x2], -1)
        if not entropy: return xn, xe
        # ------------------------------log pdf -> log of sum of each mixture component-------------------------------
        alpha = tf.nn.softmax(inputs[3], axis=2)
        alf = tf.expand_dims(alpha, 2)
        o_ = tf.expand_dims(inputs[1], 2)
        lpn = -((tf.expand_dims(xn, 3) - tf.expand_dims(inputs[0], 2)) / o_) ** 2 / 2
        lpn = tf.math.log(tf.reduce_sum(tf.exp(lpn) / o_ * alf, 3) + 1e-307) + ncn
        p = tf.expand_dims(inputs[2], 2)
        o1_ = tf.expand_dims(o1, 2)
        o2_ = tf.expand_dims(o2, 2)
        x1_ = (tf.expand_dims(x1, 3) - tf.expand_dims(u1, 2)) / o1_
        x2_ = (tf.expand_dims(x2, 3) - tf.expand_dims(u2, 2)) / o2_
        q = 1 - p ** 2
        z = -(x1_ ** 2 - 2 * p * x1_ * x2_ + x2_ ** 2) / (2 * q)
        lpe = tf.math.log(tf.reduce_sum(tf.exp(z) / (o1_ * o2_ * tf.sqrt(q)) * alf, 3) + 1e-307) + 2 * ncn
        return (xn, xe, lpn, lpe, alpha, z1, z2, o1, o2) if full_out else (xn, xe, lpn, lpe, alpha)

    @tf.function
    def entropy(self):
        """This is the node and edge's entropy contribution to the overall energy term."""
        _, _, lpn, lpe, a = self(None, full_out=False)
        return self.tw * tf.reduce_sum(lpn * self.wn * a, [1, 2, 3]) - tf.reduce_sum(lpe * self.we * a, [1, 2, 3])

    @tf.function
    def predict(self, x, bp):
        if bp is None:
            bp = trainable_variables
            a = tf.nn.softmax(bp[3], axis=2)
        else:
            a = tf.reshape(tf.nn.softmax(tf.reshape(bp[3], self.wsp), axis=3), bp[3].shape)
        u = bp[0]   # self.mu
        o = bp[1]   # self.sigma
        xn = -((x[0] - u) / o) ** 2 / 2
        # prob = tf.reduce_sum(tf.reduce_prod(tf.exp(xn) / o, 1, keepdims=True) * a, 2)
        log_con = tf.reduce_sum(xn - tf.math.log(o), axis=1, keepdims=True) + tf.math.log(a)
        max_con = tf.reduce_max(log_con, axis=2, keepdims=True)
        log_prob = tf.reshape(max_con + tf.math.log(tf.reduce_sum(tf.exp(log_con - max_con), 2, True)), [u.shape[0], -1])

        bn = tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.exp(xn) / o * a, 2) + 1e-300), 1)
        u1 = tf.gather(u, self.id1, axis=1)
        u2 = tf.gather(u, self.id2, axis=1)
        o1 = tf.gather(o, self.id1, axis=1)
        o2 = tf.gather(o, self.id2, axis=1)
        x1 = (x[1][..., 0] - u1) / o1
        x2 = (x[1][..., 1] - u2) / o2
        p = bp[2]  # self.rou
        q = 1 - p ** 2
        z = -(x1 ** 2 - 2 * p * x1 * x2 + x2 ** 2) / (2 * q)
        be = tf.math.log(tf.reduce_sum(tf.exp(z) / (o1 * o2 * tf.sqrt(q)) * a, 2) + 1e-300)
        b_ = tf.math.log(tf.reduce_sum(tf.exp(-x1 ** 2 / 2) / o1 * a, 2) *
                         tf.reduce_sum(tf.exp(-x2 ** 2 / 2) / o2 * a, 2) + 1e-300)
        be = tf.reduce_sum(be - b_, 1)
        return log_prob, bn + be

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
        bfe = -(tf.reduce_sum((fn + self.tw * lpn) * self.wn * alpha, [1, 2, 3]) +
                tf.reduce_sum((fe - lpe) * self.we * alpha, [1, 2, 3]))
        return bfe

    @tf.function
    def bfe(self, potential, belief_pool):
        a = tf.nn.softmax(tf.reshape(belief_pool[3], self.wsp), axis=3)
        alpha = tf.reshape(a, belief_pool[3].shape)
        fn, fe = potential(self(inputs=belief_pool, entropy=False))
        bfe = tf.reduce_sum(fn * self.wn * alpha, [1, 2, 3]) + tf.reduce_sum(fe * self.we * alpha, [1, 2, 3])
        return bfe / a.shape[2]

    @tf.function
    def infer(self, potential, iterations, cf=True):
        """infer the marginal distribution out by minimizing the energy of mrf, e.g bethe free energy
        Args:
            potential: potential functions of cliques, probably a neural net layer which is callable
            iterations: an integer that specify how many iterations to perform the inference
            cf: use close form method to get gradient if true otherwise tf auto-gradient
        Returns:
            stores the final energy of current state, can be used as logZ approximation
        """
        energy = tf.zeros(shape=(self.mu.shape[0]), dtype=self.dtype)
        for i in tf.range(iterations):
            if cf:
                grd, energy = self.gradient_cf(potential)
            else:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(self.trainable_variables)  # only watch mu, sigma and rou
                    energy = self.bethe_free_energy(potential)
                grd = tape.gradient(energy, self.trainable_variables)

            self.optimizer.apply_gradients(zip(grd, self.trainable_variables))

            if tf.equal(tf.math.mod(i, 50), 0) or tf.equal(i + 1, iterations):
                tf.print(tf.strings.format('iter: {} dmu = {}, dsigma = {}, drou = {}, dw={}, Energy = {}', (
                    i, tf.reduce_mean(tf.abs(grd[0])), tf.reduce_mean(tf.abs(grd[1])), tf.reduce_mean(tf.abs(grd[2])),
                    tf.reduce_mean(tf.abs(grd[3])), tf.reduce_mean(energy))))
        return energy

    @tf.function
    def gradient_cf(self, potential):
        xn, xe, lpn, lpe, alpha, z1, z2, o1, o2 = self(None)
        fn_, fe_ = potential((xn, xe))
        fn_ = (fn_ + self.tw * lpn) * self.wn
        fe_ = (fe_ - lpe) * self.we
        fn = fn_ * alpha
        fe = fe_ * alpha
        dmu = tf.reduce_sum(fn * self.xn, -1, keepdims=True) / self.sigma * sqrt2
        dsigma = tf.reduce_sum(fn * self.x22, -1, keepdims=True) / self.sigma

        q = 1 - self.rou ** 2
        sqrtq = tf.sqrt(q)
        dmu1 = tf.transpose(tf.reduce_sum(fe * (z1 - self.rou * z2), -1, keepdims=True) / (o1 * q) * sqrt2, [1, 0, 2, 3])
        dmu2 = tf.transpose(tf.reduce_sum(fe * (z2 - self.rou * z1), -1, keepdims=True) / (o2 * q) * sqrt2, [1, 0, 2, 3])
        dsigma1 = tf.transpose(tf.reduce_sum(fe * (self.xi2axj2 + self.xi2mxj2 / sqrtq), -1, keepdims=True) / o1, [1, 0, 2, 3])
        dsigma2 = tf.transpose(tf.reduce_sum(fe * (self.xi2axj2 - self.xi2mxj2 / sqrtq), -1, keepdims=True) / o2, [1, 0, 2, 3])
        drou = tf.reduce_sum(fe * (2 * self.xij - self.rou * self.xi2axj2), -1, keepdims=True) / q

        dmu += tf.transpose(tf.concat([tf.math.segment_sum(dmu1, self.id1), self.zeros], 0) +
                            tf.math.unsorted_segment_sum(dmu2, self.id2, self.N), [1, 0, 2, 3])
        dsigma += tf.transpose(tf.concat([tf.math.segment_sum(dsigma1, self.id1), self.zeros], 0) +
                               tf.math.unsorted_segment_sum(dsigma2, self.id2, self.N), [1, 0, 2, 3])
        dalpha = (tf.reduce_sum(fn_, [1, -1], keepdims=True) + tf.reduce_sum(fe_, [1, -1], keepdims=True))
        dw = alpha * (dalpha - tf.reduce_sum(dalpha * alpha, 2, keepdims=True))
        energy = -(tf.reduce_sum(fn, [1, 2, 3]) + tf.reduce_sum(fe, [1, 2, 3]))
        return (-dmu, -dsigma, -drou, -dw), energy


if __name__ == '__main__':
    from core.potential.dnn import NeuralNetPotential
    import timeit

    gmm = GMMComplete(c=10, n=10, l=1, k=11)
    pot = NeuralNetPotential(node_units=(4, 5, 5, 4), edge_units=(5, 7, 7, 5))
    print(timeit.timeit(lambda: gmm.infer(pot, 50), number=1))
