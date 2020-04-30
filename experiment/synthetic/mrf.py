# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Complete mrf model components viewed as tensorflow keras layer"""
import tensorflow as tf
from tensorflow.python.keras import Model
from numpy.polynomial.hermite import hermgauss
from core.marginal.gmm_complete import GMMComplete
from core.potential.dnn import NeuralNetPotential
from core.potential.polynominal import PolynomialPotential
from core.marginal.constants import sqrt2, pi


class CompleteMrf(Model):
    """ The complete neuralized Markov Random Field which will implement:
        Training the Markov Random Field, fit theta parameters
        Use tf.stop_gradient when doing BFE optimization like it is used in EM algor.
        Optimize Bethe Free Energy to get marginal distribution
    """

    def __init__(self, num_class, node, points, node_units, edge_units, mixture, train_rate, infer_rate, mvn, mw, ent,
                 t=10, potential='nn', dtype=tf.float64):
        super(CompleteMrf, self).__init__(name='mrf', dtype=dtype)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=train_rate)
        self.marginal = GMMComplete(c=num_class, n=node, l=mixture, k=points, lr=infer_rate, dtype=dtype)
        self.potential = NeuralNetPotential(node_units=node_units, edge_units=edge_units) if potential == 'nn' \
            else PolynomialPotential(order=2, name='order2_polynomial', dtype=dtype)
        self.logZ = tf.zeros(num_class, dtype=dtype)
        self.belief_pool = [tf.tile(v, [1, 1, t, 1]) for v in self.marginal.trainable_variables]
        self.entropy_pool = tf.tile(tf.expand_dims(self.marginal.entropy(), 1), [1, t])
        # self.Wijk, self.qp = self.quad3(mvn=mvn, k=31, dtype=dtype)
        self.mw = tf.reshape(mw, [1, 1, -1, 1])
        self.mvn_entropy = ent

    def quad3(self, mvn, k=17, dtype=tf.float64):
        """for compute 3-d integral in the most efficient way
        tri-variate quadrature method to evaluate expectation over tri-variate gaussian distribution"""
        # sp1 = [1, 1, 1, -1]
        # sp2 = [1, 1, -1, 1]
        # x, w = hermgauss(k)
        # x1, x2, x3 = tf.meshgrid(x, x, x)
        # w1, w2, w3 = tf.meshgrid(w, w, w)
        # x1 = tf.cast(tf.reshape(x1, sp1), dtype)
        # x2 = tf.cast(tf.reshape(x2, sp1), dtype)
        # x3 = tf.cast(tf.reshape(x3, sp1), dtype)
        # W_ijk = tf.cast(tf.reshape(w1 * w2 * w3 / (pi ** 1.5), sp1), dtype)
        #
        # # mvn shape = (2 X 3 X 3) now make perfect quadrature sample data points based on distribution
        # u1, u2, u3 = tf.reshape(mvn[:, 0, 0], sp2), tf.reshape(mvn[:, 0, 1], sp2), tf.reshape(mvn[:, 0, 2], sp2)
        # o1, o2, o3 = tf.reshape(mvn[:, 1, 0], sp2), tf.reshape(mvn[:, 1, 1], sp2), tf.reshape(mvn[:, 1, 2], sp2)
        # p1, p2, p3 = tf.reshape(mvn[:, 2, 0], sp2), tf.reshape(mvn[:, 2, 1], sp2), tf.reshape(mvn[:, 2, 2], sp2)
        # sqt1mp1s = tf.sqrt(1 - p1 ** 2)
        # sqtdta = tf.sqrt(1 + 2 * p1 * p2 * p3 - p1 ** 2 - p2 ** 2 - p3 ** 2)
        #
        # X1 = sqrt2 * o1 * x1 + u1
        # X2 = sqrt2 * o2 * (p1 * x1 + sqt1mp1s * x2) + u2
        # X3 = sqrt2 * o3 * (p2 * x1 + (p3 - p1 * p2) / sqt1mp1s * x2 + sqtdta / sqt1mp1s * x3) + u3
        # Xn = tf.concat([X1, X2, X3], axis=1)
        # Xe = tf.stack([tf.gather(Xn, self.marginal.id1, axis=1), tf.gather(Xn, self.marginal.id2, axis=1)], -1)
        # return W_ijk, (Xn, Xe)
        # return W_ijk, (tf.concat([X1, sqrt2 * o2 * x2 + u2, sqrt2 * o3 * x3 + u3], axis=1), Xe)

    def call(self, inputs):
        """compute the join log probability of all classes on this mrf for the input data (batch)
        the input (node, edge) shape is like: node = (1, M, N, 1, L, batch_size,), L=1
        edge = (1, M, N, 2, L, batch_size, 2), it works because batch tf.matmul(input,kernel) support broadcasting
        return shape = (C, batch_size)
        """
        pn, pe = self.potential(inputs)
        scores = tf.reduce_sum(pn, [1, 2]) + tf.reduce_sum(pe, [1, 2])
        log_prob = scores - tf.expand_dims(self.logZ, -1)
        return log_prob, scores

    def mung(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data).map(lambda x: (
            x, tf.stack([tf.gather(x, self.marginal.id1, axis=1), tf.gather(x, self.marginal.id2, axis=1)], -1)))
        # size, c, n = data.shape
        # xn = tf.stack([d[0] for d in dataset])
        # xe = tf.stack([d[1] for d in dataset])
        # xn, xe = (tf.expand_dims(tf.transpose(xn, [1, 2, 0]), -2), tf.expand_dims(tf.transpose(xe, [1, 2, 0, 3]), -3))
        # y_true = tf.reshape(tf.tile(tf.expand_dims(tf.range(c), -1), (1, size)), (c * size,))
        # xn = tf.reshape(tf.transpose(xn, (1, 2, 0, 3)), (1, n, 1, -1))
        # xe = tf.reshape(tf.transpose(xe, (1, 2, 0, 3, 4)), (1, xe.shape[1], 1, -1, 2))
        return dataset  # , xn, xe, y_true

    def train(self, data, valid_size, infer_iter, batch_size, epochs, log_dir):
        """marginal inference using gradient ascent to get best fit of theta which parameterized
         the potential neural nets, as well as those (u, o, p), which defines our underling
         marginal distribution
        """
        train_set = self.mung(data[valid_size:, ...])
        summary_writer = tf.summary.create_file_writer(log_dir)
        counter = 0
        for e in tf.range(epochs):
            tds = train_set.shuffle(data.shape[0] - valid_size).batch(batch_size)
            for itr, x_batch_train in enumerate(tds):
                x_batch_train = (tf.expand_dims(tf.transpose(x_batch_train[0], [1, 2, 0]), -2),
                                 tf.expand_dims(tf.transpose(x_batch_train[1], [1, 2, 0, 3]), -3))

                """step 1: infer the best marginal distribution under current potential state"""
                # if counter == 0: tf.summary.trace_on(graph=True, profiler=True)
                with tf.control_dependencies(self.potential.trainable_variables):
                    self.marginal.init_marginal()  # otherwise not stable?
                    self.marginal.infer(potential=self.potential, iterations=infer_iter, cf=True)

                """step 1.5 update belief pool"""
                self.entropy_pool = tf.concat([self.entropy_pool[:, 1:], tf.expand_dims(self.marginal.entropy(), 1)], 1)
                self.belief_pool = [tf.concat([p[:, :, v.shape[2]:, :], v], 2) for p, v in
                                    zip(self.belief_pool, self.marginal.trainable_variables)]

                """step 2: get the expected value of each potential's gradient over theta with the marginal"""
                # if tf.equal(total_iter, 0): tf.summary.trace_on(graph=True, profiler=True)
                # using b_bar to update(train) model parameter
                with tf.control_dependencies([self.potential.trainable_variables, self.belief_pool, self.entropy_pool]):
                    neg_lld, self.logZ = self.train_one_step(x_batch_train)

                    """step 3: show the goodness of this generative model so far thru label prediction accuracy"""
                    kld = self.kl_divergence()
                    lld = -tf.reduce_mean(neg_lld)

                """step 4: print and record those result for analysis"""
                tf.print(tf.strings.format('(epoch)steps: ({}){}, kld = {}, lld = {}', (e, itr, kld, lld)))
                with summary_writer.as_default():
                    tf.summary.scalar('kl_divergence', kld, step=counter)
                    tf.summary.scalar('log-likelihood', lld, step=counter)
                    # if counter == 0: tf.summary.trace_export(name="marginal_inference", step=0, profiler_outdir='.')
                counter += 1
            # if tf.equal(tf.math.mod(e, 50), 0):
            #     self.save_weights('saved_model', save_format='tf')

    def train_one_step(self, x_batch):
        """training one step of mrf based on current state of marginal distribution"""
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.potential.trainable_variables)
            fn, fe = self.potential(x_batch)
            bfe = self.marginal.bfe(self.potential, self.belief_pool) + tf.reduce_mean(self.entropy_pool, 1)
            neg_lld = bfe - tf.reduce_mean(tf.reduce_sum(fn, [1, 2]) + tf.reduce_sum(fe, [1, 2]), -1)
        grd = tape.gradient(neg_lld, self.potential.trainable_variables)
        self.optimizer.apply_gradients(zip(grd, self.potential.trainable_variables))
        return neg_lld, bfe

    def predict_label(self, x):
        """Note x has different shape when is validation data or test data
        x_train[0]: (C, n, 1, batch_size) x_train[1]: (C, e, 1, batch_size,2)
        x_test[0]: (1, n, 1, batch_size)  x_test[1]: (1, e, 1, batch_size,2)
        """
        p, p2 = self(x)
        y_pred, y_pred2 = tf.argmax(p), tf.argmax(p2)
        p3, p4 = self.marginal.predict(x, bp=self.belief_pool)
        y_pred3, y_pred4 = tf.argmax(p3), tf.argmax(p4)
        return y_pred, y_pred2, y_pred3, y_pred4

    def kl_divergence(self):
        fn, fe = self.potential(self.qp)
        kld = -self.mvn_entropy - tf.squeeze(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(fn + fe, 1, True) * self.Wijk,
                                                                         3, True) * self.mw, 2)) + self.logZ[0]
        return kld
