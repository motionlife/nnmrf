# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Grid mrf model components viewed as tensorflow keras layer"""

import tensorflow as tf
from tensorflow.python.keras import Model

from core.marginal.gmm_grid import GMMGrid, GMMGrid0
from core.potential.dnn import NeuralNetPotential
from core.potential.polynominal import PolynomialPotential


def mung(train_ds, valid_ds):
    x_test, y_test = valid_ds
    _, m, n, c = train_ds.shape
    train_dataset = tf.data.Dataset.from_tensor_slices(train_ds).map(
        lambda x: tf.reshape(x, [m, n, 1, 1, c])).map(
        lambda x: (x, tf.stack([tf.tile(x, [1, 1, 2, 1, 1]),
                                tf.concat([tf.roll(x, -1, 0), tf.roll(x, -1, 1)], 2)], -1)))

    x_test = tf.reshape(tf.transpose(x_test, [1, 2, 0]), [1, m, n, 1, 1, -1])
    valid_data = (x_test, tf.stack(
        [tf.tile(x_test, [1, 1, 1, 2, 1, 1]), tf.concat([tf.roll(x_test, -1, 1), tf.roll(x_test, -1, 2)], 3)],
        -1)), y_test
    return train_dataset, valid_data


class GridMrf(Model):
    """ The complete neuralized Markov Random Field which will implement:
        Training the Markov Random Field, fit theta parameters
        Use tf.stop_gradient when doing BFE optimization like it is used in EM algor.
        Optimize Bethe Free Energy to get marginal distribution
    """

    def __init__(self, num_class, height, width, points, node_units, edge_units, mixture, train_rate, infer_rate,
                 potential='nn', dtype=tf.float64):
        super(GridMrf, self).__init__(name='mrf', dtype=dtype)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=train_rate)
        self.marginal = GMMGrid0(c=num_class, m=height, n=width, l=mixture, k=points, lr=infer_rate,
                                 dtype=dtype)
        self.potential = NeuralNetPotential(node_units=node_units, edge_units=edge_units) if potential == 'nn' \
            else PolynomialPotential(order=2, name='order2_polynomial', dtype=dtype)
        self.logZ = tf.zeros((num_class,), dtype=dtype)

    @tf.function
    def call(self, inputs):
        """compute the join log probability of all classes on this mrf for the input data (batch)
        the input (node, edge) shape is like: node = (1, M, N, 1, L, batch_size,), L=1
        edge = (1, M, N, 2, L, batch_size, 2), it works because batch tf.matmul(input,kernel) support broadcasting
        return shape = (C, batch_size)
        """
        pn, pe = self.potential(inputs)
        scores = tf.reduce_sum(pn, [1, 2, 3, 4]) + tf.reduce_sum(pe, [1, 2, 3, 4])
        log_prob = scores - tf.expand_dims(self.logZ, -1)
        return log_prob, scores

    def train(self, train_data, valid_data, log_dir, infer_iter, batch_size, epochs):
        """marginal inference using gradient ascent to get best fit of theta which parameterized
         the potential neural nets, as well as those (u, o, p), which defines our underling
         marginal distribution
        """
        train_dataset, valid_data = mung(train_data, valid_data)
        train_metric = tf.keras.metrics.Accuracy(name="train_accuracy")
        train_metric2 = tf.keras.metrics.Accuracy(name="train_accuracy2")
        train_metric3 = tf.keras.metrics.Accuracy(name="train_accuracy3")
        valid_metric = tf.keras.metrics.Accuracy(name="valid_accuracy")
        valid_metric2 = tf.keras.metrics.Accuracy(name="valid_accuracy2")
        summary_writer = tf.summary.create_file_writer(log_dir)
        counter = 0
        for e in tf.range(epochs):
            tds = train_dataset.shuffle(6750).batch(batch_size).prefetch(32)
            for itr, x_batch_train in enumerate(tds):
                x_batch_train = tf.transpose(x_batch_train[0], [5, 1, 2, 3, 4, 0]), tf.transpose(x_batch_train[1],
                                                                                                 [5, 1, 2, 3, 4, 0, 6])
                """step 1: infer the best marginal distribution under current potential state"""
                self.marginal.init_marginal()  # otherwise not stable?
                # if tf.equal(total_iter, 0): tf.summary.trace_on(graph=True, profiler=True)
                self.logZ = -self.marginal.infer(potential=self.potential, iterations=infer_iter)

                """step 2: get the expected value of each potential's gradient over theta with the marginal"""
                # if tf.equal(total_iter, 0): tf.summary.trace_on(graph=True, profiler=True)
                neg_log_likelihood = self.train_one_step(x_batch_train)

                """step 3: show the goodness of this generative model so far thru label prediction accuracy"""
                ty_true, ty_pred, ty_pred2, ty_pred3 = self.predict_label(x_batch_train, x_is_train=True)
                train_metric.update_state(ty_true, ty_pred)
                train_metric2.update_state(ty_true, ty_pred2)
                train_metric3.update_state(ty_true, ty_pred3)
                vy_pred, vy_pred2 = self.predict_label(valid_data[0])
                valid_metric.update_state(valid_data[1], vy_pred)
                valid_metric2.update_state(valid_data[1], vy_pred2)
                train_acc = train_metric.result()
                train_acc2 = train_metric2.result()
                train_acc3 = train_metric3.result()
                valid_acc = valid_metric.result()
                valid_acc2 = valid_metric2.result()
                lld = -tf.reduce_mean(neg_log_likelihood)

                """step 4: print and record those result for analysis"""
                tf.print(tf.strings.format(
                    '(epoch)steps: ({}){}, train_acc = {}, train_acc3 = {}, valid_acc = {}, lld = {}-----',
                    (e, itr, train_acc, train_acc3, valid_acc, lld)))
                with summary_writer.as_default():
                    tf.summary.scalar('train_accuracy', tf.maximum(train_acc, train_acc2, train_acc3), step=counter)
                    tf.summary.scalar('valid_accuracy', tf.maximum(valid_acc, valid_acc2), step=counter)
                    tf.summary.scalar('train_accuracy3', train_acc3, step=counter)
                    tf.summary.scalar('log-likelihood', lld, step=counter)
                    # tf.summary.trace_export(name="marginal_inference", step=0, profiler_outdir=log_dir)
                counter += 1
                valid_metric.reset_states()
                valid_metric2.reset_states()
                train_metric.reset_states()
                train_metric2.reset_states()
                train_metric3.reset_states()
            # if tf.equal(tf.math.mod(e, 50), 0):
            #     self.save_weights('saved_model', save_format='tf')

    # @tf.function
    def train_one_step(self, x_batch):
        """training one step of mrf based on current state of marginal distribution"""
        bs = x_batch[0].shape[5]  # batch size of data
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.potential.trainable_variables)
            fn, fe = self.potential(x_batch)
            neg_log_likelihood = -(tf.reduce_sum(fn, [1, 2, 3, 4, 5]) / bs + tf.reduce_sum(fe, [1, 2, 3, 4, 5]) / bs
                                   + self.marginal.bethe_free_energy(self.potential))
        grd = tape.gradient(neg_log_likelihood, self.potential.trainable_variables)
        self.optimizer.apply_gradients(zip(grd, self.potential.trainable_variables))
        return neg_log_likelihood

    def predict_label(self, x, x_is_train=False):
        """Note x has different shape when is validation data or test data
        x_train[0]: (C, M, N, 1, 1, batch_size) x_train[1]: (C, M, N, 2, 1, batch_size,2)
        x_test[0]: (1, M, N, 1, 1, batch_size)  x_test[1]: (1, M, N, 2, 1, batch_size,2)
        """
        if x_is_train:
            shp = x[0].shape
            c, m, n, l, bs = shp[0], shp[1], shp[2], shp[4], shp[5]
            xn = tf.reshape(tf.transpose(x[0], (1, 2, 3, 4, 0, 5)), (1, m, n, 1, 1, -1))
            xe = tf.reshape(tf.transpose(x[1], (1, 2, 3, 4, 0, 5, 6)), (1, m, n, 2, 1, -1, 2))
            p, p2 = self((xn, xe))
            y_pred, y_pred2 = tf.argmax(p), tf.argmax(p2)
            y_true = tf.reshape(tf.tile(tf.expand_dims(tf.range(c), -1), (1, bs)), y_pred.shape)
            return y_true, y_pred, y_pred2, tf.argmax(self.marginal.eval(xn, xe))
        y_pred, y_pred2 = self(x)
        return tf.argmax(y_pred), tf.argmax(y_pred2)


if __name__ == '__main__':
    print('---test all the functionality in this module---')
    # import timeit

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # marginal = Marginal(10, 28, 28, K=17, dtype=tf.float64)
    # potential = Potential([3, 4, 3], [4, 5, 4])
    # mrf = MRF(node_units=(3, 4, 3), edge_units=(4, 5, 4))
    # mrf.fit()
    # compare the efficiency of GPU with CPU (10x faster on P5000 than intel gold 6130)
    # print(timeit.timeit(lambda: mrf.train(), number=100))
