import tensorflow as tf
from core.marginal.gmm_hidden import GMMHidden
from core.potential.dnn import NeuralNetPotential
from core.potential.polynominal import PolynomialPotential

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class GmmUnitTest(tf.test.TestCase):

    def testGradient(self):
        """test if close form gradient consistent with tensorflow's auto-gradient result"""
        tf.random.set_seed(7)
        c = 4
        l = 4
        bs = 2
        gmm = GMMHidden(c=c, m=27, l=l, k=51, bs=bs, dtype=tf.float64)
        # u = tf.random.uniform([c, m, n, 1, 1, 1], -3, 3, dtype=gmm.dtype)
        # o = tf.random.uniform([c, m, n, 1, 1, 1], 0.001, 10, dtype=gmm.dtype)
        # p = tf.random.uniform((c, m, n, 2, 1, 1), -0.999, 0.999, dtype=gmm.dtype)
        # gmm.mu.assign(tf.tile(u, [1, 1, 1, 1, l, 1]))
        # gmm.sigma.assign(tf.tile(o, [1, 1, 1, 1, l, 1]))
        # gmm.rou.assign(tf.tile(p, [1, 1, 1, 1, l, 1]))

        gmm.mu.assign(tf.random.uniform(gmm.mu.get_shape(), 0, 1, dtype=gmm.dtype))
        gmm.sigma.assign(tf.random.uniform(gmm.sigma.get_shape(), 0.8, 1, dtype=gmm.dtype))
        # gmm.rou.assign(tf.random.uniform(gmm.rou.get_shape(), -0.99, 0.99, dtype=gmm.dtype))
        gmm.w.assign(tf.random.uniform(gmm.w.get_shape(), -7, 7, dtype=gmm.dtype))

        # pot = PolynomialPotential(c=c, m=27, n=27, order=2, dtype=tf.float64)
        pot = NeuralNetPotential(node_units=(3, 4, 4, 3), edge_units=(4, 5, 5, 4), dtype=tf.float64)

        x0 = tf.random.uniform([c, bs, 729, 1, 1], 0, 1, dtype=gmm.dtype)
        x0 = tf.tile(x0, [1, 1, 1, l, 1])
        gmm.mu[:, 1:, :729, ...].assign(x0)
        gmm.sigma[:, 1:, :729, ...].assign(tf.zeros_like(x0))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(gmm.trainable_variables)
            bfe = gmm.bethe_free_energy(pot)
        grd = tape.gradient(bfe, gmm.trainable_variables)
        grd = [tf.where(tf.math.is_nan(grd[i]), tf.zeros_like(grd[i]), grd[i]) for i in range(len(grd))]
        # grd, bfe = gmm.gradient_cf(pot)

        # --------------------below are close form gradient calculating---------------------
        gmm2 = GMMHidden(c=c, m=27, l=l, k=17, bs=bs, dtype=tf.float64)
        gmm2.mu.assign(gmm.mu.read_value())
        gmm2.sigma.assign(gmm.sigma.read_value())
        # gmm2.rou.assign(gmm.rou.read_value())
        gmm2.w.assign(gmm.w.read_value())
        # with tf.GradientTape(watch_accessed_variables=False) as tape:
        #     tape.watch(gmm2.trainable_variables)
        #     bfe2 = gmm2.bethe_free_energy(pot)
        # grd2 = tape.gradient(bfe2, gmm2.trainable_variables)
        grd2, bfe2 = gmm2.gradient_cf(pot)

        tolerance = 1e-6
        # self.assertAllClose(bfe, bfe2, tolerance, tolerance)
        self.assertAllClose(grd[:-1], grd2[:-1], rtol=tolerance, atol=tolerance)
        print(f'max grd{[tf.reduce_max(i).numpy() for i in grd]}')
        print(f'max grd2{[tf.reduce_max(i).numpy() for i in grd2]}')
        print(f'min grd{[tf.reduce_min(i).numpy() for i in grd]}')
        print(f'min grd2{[tf.reduce_min(i).numpy() for i in grd2]}')


if __name__ == '__main__':
    tf.test.main()  # run all unit tests
