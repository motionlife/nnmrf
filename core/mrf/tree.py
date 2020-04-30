# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Tree mrf model components viewed as tensorflow keras layer"""

import tensorflow as tf
from tensorflow.python.keras import Model

from core.marginal.gmm_hidden import GMMHidden
from core.potential.dnn import NeuralNetPotential
from core.potential.polynominal import PolynomialPotential


class TreeMrf(Model):
    def __init__(self, num_class, img_width, points, node_units, edge_units, mixture, train_rate, infer_rate,
                 batch_size, potential='nn', dtype=tf.float64):
        super(TreeMrf, self).__init__(name='tree_mrf', dtype=dtype)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=train_rate)
        self.marginal = GMMHidden(c=num_class, m=img_width, l=mixture, bs=batch_size, k=points, lr=infer_rate)
        self.potential = NeuralNetPotential(node_units=node_units, edge_units=edge_units) if potential == 'nn' \
            else PolynomialPotential(c=num_class, m=img_width, n=img_width, order=2, name='order2_polynomial')
        self.logZ = tf.zeros((num_class, 1 + batch_size), dtype=dtype)

    @tf.function
    def call(self, inputs):
        pass

    def train(self, train_data, valid_data, log_dir, infer_iter, batch_size, epochs):
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data).map(lambda x: tf.reshape(x, [-1, 10, 1, 1]))
        valid_x, valid_y = valid_data
        train_metric = tf.keras.metrics.Accuracy(name="train_accuracy")
        train_metric1 = tf.keras.metrics.Accuracy(name="train_accuracy1")
        valid_metric = tf.keras.metrics.Accuracy(name="valid_accuracy")
        valid_metric1 = tf.keras.metrics.Accuracy(name="valid_accuracy1")
        summary_writer = tf.summary.create_file_writer(log_dir)
        counter = 0
        for e in tf.range(epochs):
            tds = train_dataset.shuffle(6744).batch(batch_size).prefetch(32)
            for itr, x_batch_train in enumerate(tds):
                train_y = tf.reshape(tf.tile(tf.expand_dims(tf.range(10), -1), (1, x_batch_train.shape[0])), [-1])
                x_batch_train = tf.transpose(x_batch_train, [2, 0, 1, 3, 4])  # [c, bs, node, 1, 1]

                # inference thd distribution
                self.marginal.init_marginal()
                self.logZ = - self.marginal.infer(self.potential, infer_iter, x0=x_batch_train)

                # train potential parameters
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(self.potential.trainable_variables)
                    ept = self.marginal.potential_expectation(self.potential)
                    loss = ept[:, 0] - tf.reduce_mean(ept[:, 1:], 1)
                grd = tape.gradient(loss, self.potential.trainable_variables)
                self.optimizer.apply_gradients(zip(grd, self.potential.trainable_variables))

                # record performance, e.g. lld and prediction accuracy
                lld = tf.reduce_mean(tf.reduce_mean(self.logZ[:, 1:], 1) - self.logZ[:, 0])
                tp0, tp1 = self.marginal.predict(tf.reshape(x_batch_train, [1, -1, 729, 1, 1]))
                ty_pred0, ty_pred1 = tf.argmax(tp0), tf.argmax(tp1)
                vp0, vp1 = self.marginal.predict(tf.reshape(valid_x, [1, -1, 729, 1, 1]))
                vy_pred0, vy_pred1 = tf.argmax(vp0), tf.argmax(vp1)
                train_metric.update_state(train_y, ty_pred0)
                train_metric1.update_state(train_y, ty_pred1)
                valid_metric.update_state(valid_y, vy_pred0)
                valid_metric1.update_state(valid_y, vy_pred1)
                train_acc = train_metric.result()
                train_acc1 = train_metric1.result()
                valid_acc = valid_metric.result()
                valid_acc1 = valid_metric1.result()

                tf.print(tf.strings.format(
                    '(epoch)steps: ({}){}, train_acc = {}, train_acc1 = {}, valid_acc = {}, valid_acc1 = {}, lld = {}',
                    (e, itr, train_acc, train_acc1, valid_acc, valid_acc1, lld)))
                with summary_writer.as_default():
                    tf.summary.scalar('train_accuracy', tf.maximum(train_acc, train_acc1), step=counter)
                    tf.summary.scalar('valid_accuracy', tf.maximum(valid_acc, valid_acc1), step=counter)
                    tf.summary.scalar('log-likelihood', lld, step=counter)
                counter += 1
                valid_metric.reset_states()
                valid_metric1.reset_states()
                train_metric.reset_states()
                train_metric1.reset_states()
            # if tf.equal(tf.math.mod(e, 50), 0):
            #     self.save_weights('saved_model', save_format='tf')
