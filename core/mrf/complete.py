# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Complete mrf model components viewed as tensorflow keras layer"""

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Model

from core.marginal.gmm_complete import GMMComplete
from core.potential.dnn import NeuralNetPotential
from core.potential.polynominal import PolynomialPotential


class CompleteMrf(Model):
    """ The complete neuralized Markov Random Field which will implement:
        Training the Markov Random Field, fit theta parameters
        Use tf.stop_gradient when doing BFE optimization like it is used in EM algor.
        Optimize Bethe Free Energy to get marginal distribution
    """

    def __init__(self, num_class, node, points, node_units, edge_units, mixture, train_rate, infer_rate, t=10,
                 potential='nn', dtype=tf.float64):
        super(CompleteMrf, self).__init__(name='mrf', dtype=dtype)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=train_rate)
        self.marginal = GMMComplete(c=num_class, n=node, l=mixture, k=points, lr=infer_rate, dtype=dtype)
        self.potential = NeuralNetPotential(node_units=node_units, edge_units=edge_units) if potential == 'nn' \
            else PolynomialPotential(c=num_class, m=node, n=node, order=2, name='order2_polynomial', dtype=dtype)
        self.logZ = tf.zeros(num_class, dtype=dtype)
        self.belief_pool = [tf.tile(v, [1, 1, t, 1]) for v in self.marginal.trainable_variables]
        self.entropy_pool = tf.tile(tf.expand_dims(self.marginal.entropy(), 1), [1, t])

    @tf.function
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

        size, c, n = data.shape
        xn = tf.stack([d[0] for d in dataset])
        xe = tf.stack([d[1] for d in dataset])
        xn, xe = (tf.expand_dims(tf.transpose(xn, [1, 2, 0]), -2), tf.expand_dims(tf.transpose(xe, [1, 2, 0, 3]), -3))
        y_true = tf.reshape(tf.tile(tf.expand_dims(tf.range(c), -1), (1, size)), (c * size,))
        xn = tf.reshape(tf.transpose(xn, (1, 2, 0, 3)), (1, n, 1, -1))
        xe = tf.reshape(tf.transpose(xe, (1, 2, 0, 3, 4)), (1, xe.shape[1], 1, -1, 2))
        return dataset, xn, xe, y_true

    def train(self, data, valid_size, infer_iter, batch_size, epochs, log_dir):
        """marginal inference using gradient ascent to get best fit of theta which parameterized
         the potential neural nets, as well as those (u, o, p), which defines our underling
         marginal distribution
        """
        train_set, xnt, xet, yt_true = self.mung(data[valid_size:, ...])
        _, xnv, xev, yv_true = self.mung(data[:valid_size, ...])

        train_metric = tf.keras.metrics.Accuracy(name="train_accuracy")
        train_metric2 = tf.keras.metrics.Accuracy(name="train_accuracy2")
        train_metric3 = tf.keras.metrics.Accuracy(name="train_accuracy3")
        train_metric4 = tf.keras.metrics.Accuracy(name="train_accuracy3")

        valid_metric = tf.keras.metrics.Accuracy(name="valid_accuracy")
        valid_metric2 = tf.keras.metrics.Accuracy(name="valid_accuracy2")
        valid_metric3 = tf.keras.metrics.Accuracy(name="valid_accuracy3")
        valid_metric4 = tf.keras.metrics.Accuracy(name="valid_accuracy3")

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
                    ty_pred, ty_pred2, ty_pred3, ty_pred4 = self.predict_label((xnt, xet), b=500)
                    train_metric.update_state(yt_true, ty_pred)
                    train_metric2.update_state(yt_true, ty_pred2)
                    train_metric3.update_state(yt_true, ty_pred3)
                    train_metric4.update_state(yt_true, ty_pred4)
                    train_acc = train_metric.result()
                    train_acc2 = train_metric2.result()
                    train_acc3 = train_metric3.result()
                    train_acc4 = train_metric4.result()
                    lld = -tf.reduce_mean(neg_lld)

                    vy_pred, vy_pred2, vy_pred3, vy_pred4 = self.predict_label((xnv, xev), b=500)
                    valid_metric.update_state(yv_true, vy_pred)
                    valid_metric2.update_state(yv_true, vy_pred2)
                    valid_metric3.update_state(yv_true, vy_pred3)
                    valid_metric4.update_state(yv_true, vy_pred4)
                    valid_acc = valid_metric.result()
                    valid_acc2 = valid_metric2.result()
                    valid_acc3 = valid_metric3.result()
                    valid_acc4 = valid_metric4.result()

                """step 4: print and record those result for analysis"""
                tf.print(tf.strings.format(
                    '(epoch)steps: ({}){}, train_acc = {}, train_acc2 = {}, train_acc3 = {}, train_acc4 = {}, lld = {}',
                    (e, itr, train_acc, train_acc2, train_acc3, train_acc4, lld)))
                with summary_writer.as_default():
                    tf.summary.scalar('train_accuracy', tf.maximum(train_acc, train_acc2), step=counter)
                    tf.summary.scalar('valid_accuracy', tf.maximum(valid_acc, valid_acc2), step=counter)
                    tf.summary.scalar('train_accuracy_MarginalInfer', tf.maximum(train_acc3, train_acc4), step=counter)
                    tf.summary.scalar('valid_accuracy_MarginalInfer', tf.maximum(valid_acc3, valid_acc4), step=counter)
                    tf.summary.scalar('log-likelihood', lld, step=counter)
                    # if counter == 0: tf.summary.trace_export(name="marginal_inference", step=0, profiler_outdir='.')
                counter += 1
                train_metric.reset_states()
                train_metric2.reset_states()
                train_metric3.reset_states()
                train_metric4.reset_states()
                valid_metric.reset_states()
                valid_metric2.reset_states()
                valid_metric3.reset_states()
                valid_metric4.reset_states()
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

    def predict_label(self, x, b=None):
        """Note x has different shape when is validation data or test data
        x_train[0]: (C, n, 1, batch_size) x_train[1]: (C, e, 1, batch_size,2)
        x_test[0]: (1, n, 1, batch_size)  x_test[1]: (1, e, 1, batch_size,2)
        """
        if b is None:
            p, p2 = self(x)
            y_pred, y_pred2 = tf.argmax(p), tf.argmax(p2)
            p3, p4 = self.marginal.predict(x, bp=self.belief_pool)
            y_pred3, y_pred4 = tf.argmax(p3), tf.argmax(p4)
            return y_pred, y_pred2, y_pred3, y_pred4
        else:
            # do batch prediction
            size = x[0].shape[-1]
            result = [np.zeros(shape=[size]), np.zeros(shape=[size]), np.zeros(shape=[size]), np.zeros(shape=[size])]
            q, r = divmod(size, b)
            for i in tf.range(q):
                start = i * b
                end = start + b
                for j, y in enumerate(self.predict_label((x[0][..., start:end], x[1][..., start:end, :]))):
                    result[j][start:end] = y
            if r > 0:
                start = q * b
                end = start + r
                for j, y in enumerate(self.predict_label((x[0][..., start:end], x[1][..., start:end, :]))):
                    result[j][start:end] = y

            return result
