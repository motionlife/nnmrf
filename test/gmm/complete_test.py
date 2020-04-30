import tensorflow as tf
from core.marginal.gmm_complete import GMMComplete
from core.potential.dnn import NeuralNetPotential
from core.potential.polynominal import PolynomialPotential

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class GmmUnitTest(tf.test.TestCase):

    def testMixtureNodeEntropy(self):
        """test node mixture entropy to see if it's consistent with matlab result"""
        gmm = GMMComplete(c=1, n=1, l=4, k=11, dtype=tf.float64)
        mu = tf.reshape(tf.constant([4.292636231872278, -1.500162340151913, -3.034047495687918, -2.489161420239689],
                                    dtype=gmm.dtype), gmm.mu.get_shape())
        sigma = tf.reshape(tf.constant([3.080223380733196, 2.366444244513646, 1.758297535314984, 4.154143139481454],
                                       dtype=gmm.dtype), gmm.sigma.get_shape())
        n_entropy = tf.constant(2.808113860036801, dtype=gmm.dtype)
        gmm.mu.assign(mu)
        gmm.sigma.assign(sigma)
        _, _, lpn, _, _ = gmm(None, full_out=False)
        hn = -tf.reduce_sum(lpn * gmm.wn) / 4
        tolerance = 1e-12
        self.assertAllClose(n_entropy, hn, rtol=tolerance, atol=tolerance)
        print(f'node entropy from matlab: {n_entropy}')
        print(f'node entropy from tensorflow: {hn}')

    def testMixtureEdgeEntropy(self):
        """test if edge mixture entropy to see if it's consistent with matlab result"""
        gmm = GMMComplete(c=1, n=2, l=4, k=11, dtype=tf.float64)
        u1 = tf.constant([0.852640911527243, 0.497236082911395, 4.171936638298100, -2.141609811796265], dtype=gmm.dtype)
        u2 = tf.constant([-4.241457104369363, -4.460498813333929, 0.307975530089727, 2.791672301020112],
                         dtype=gmm.dtype)
        o1 = tf.constant([7.572002291107213, 7.537290942784953, 3.804458469753567, 5.678216407252211], dtype=gmm.dtype)
        o2 = tf.constant([9.340106842291830, 1.299062084737301, 5.688236608721927, 4.693906410582058], dtype=gmm.dtype)
        p = tf.constant([-0.976195860997517, -0.325754711202237, -0.675635383613515, 0.588569081367814],
                        dtype=gmm.dtype)
        e_entropy = tf.constant(6.269906108401816, dtype=gmm.dtype)

        gmm.mu.assign(tf.reshape(tf.stack([u1, u2]), gmm.mu.get_shape()))
        gmm.sigma.assign(tf.reshape(tf.stack([o1, o2]), gmm.sigma.get_shape()))
        gmm.rou[0, 0, :, 0].assign(p)
        _, _, _, lpe, _ = gmm(None, full_out=False)
        he = -tf.reduce_sum((lpe * gmm.we)[0, 0, :, :]) / 4
        tolerance = 1e-12
        self.assertAllClose(e_entropy, he, rtol=tolerance, atol=tolerance)
        print(f'edge entropy from matlab: {e_entropy}')
        print(f'edge entropy from tensorflow: {he}')

    def testGradient(self):
        """test if close form gradient consistent with tensorflow's auto-gradient result"""
        tf.random.set_seed(9)
        c = 5
        n = 10
        l = 4
        gmm = GMMComplete(c=c, n=n, l=l, k=99, dtype=tf.float64)

        # u = tf.random.uniform([c, n, 1, 1], -5, 5, dtype=gmm.dtype)
        # o = tf.random.uniform([c, n, 1, 1], 0.001, 10, dtype=gmm.dtype)
        # p = tf.random.uniform([c, int(n * (n - 1) / 2), 1, 1], -0.999, 0.999, dtype=gmm.dtype)
        # gmm.mu.assign(tf.tile(u, [1, 1, l, 1]))
        # gmm.sigma.assign(tf.tile(o, [1, 1, l, 1]))
        # gmm.rou.assign(tf.tile(p, [1, 1, l, 1]))

        gmm.mu.assign(tf.random.uniform(gmm.mu.get_shape(), 0, 1, dtype=gmm.dtype))
        gmm.sigma.assign(tf.random.uniform(gmm.sigma.get_shape(), 0.5, 1, dtype=gmm.dtype))
        gmm.rou.assign(tf.random.uniform(gmm.rou.get_shape(), -0.5, 0.5, dtype=gmm.dtype))
        gmm.w.assign(tf.random.uniform(gmm.w.get_shape(), -3, 3, dtype=gmm.dtype))

        pot = PolynomialPotential(order=2, dtype=tf.float64)
        # pot = NeuralNetPotential(node_units=(3, 4, 4, 3), edge_units=(4, 5, 5, 4), dtype=tf.float64)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(gmm.trainable_variables)
            bfe = gmm.bethe_free_energy(pot)
        grd = tape.gradient(bfe, gmm.trainable_variables)
        # grd, bfe = gmm.gradient_cf(pot)

        # --------------------below are close form gradient calculating---------------------
        gmm2 = GMMComplete(c=c, n=n, l=l, k=51, dtype=tf.float64)
        for i, v in enumerate(gmm2.trainable_variables):
            v.assign(gmm.trainable_variables[i].read_value())
        grd2, bfe2 = gmm2.gradient_cf(pot)

        tolerance = 1e-2
        # self.assertAllClose(bfe, bfe2, tolerance, tolerance)
        self.assertAllClose(grd[1], grd2[1], rtol=tolerance, atol=tolerance)

        print(f'max auto{[tf.reduce_max(i).numpy() for i in grd]}')
        print(f'pos{[tf.reduce_sum(tf.argmax(i)).numpy() for i in grd]}')

        print(f'max clsf{[tf.reduce_max(i).numpy() for i in grd2]}')
        print(f'pos{[tf.reduce_sum(tf.argmax(i)).numpy() for i in grd2]}')

        print(f'min auto{[tf.reduce_min(i).numpy() for i in grd]}')
        print(f'pos{[tf.reduce_sum(tf.argmin(i)).numpy() for i in grd]}')

        print(f'min clsf{[tf.reduce_min(i).numpy() for i in grd2]}')
        print(f'pos{[tf.reduce_sum(tf.argmin(i)).numpy() for i in grd2]}')
        # print(f'energy is: {bfe}')
        # for v in gmm.trainable_variables:
        #     print(tf.squeeze(v))

    def testSanity2D(self):
        gmm = GMMComplete(c=1, n=2, l=2, k=17, dtype=tf.float64)
        u1 = tf.constant([-4.95, 2.55], dtype=gmm.dtype)
        u2 = tf.constant([3.22, -3.47], dtype=gmm.dtype)
        o1 = tf.constant([2.35, 0.17], dtype=gmm.dtype)
        o2 = tf.constant([0.05, 1.29], dtype=gmm.dtype)
        p = tf.constant([-0.5, 0.4], dtype=gmm.dtype)
        w = tf.constant([3, 4], dtype=gmm.dtype)
        e_entropy = tf.constant(1.632528390162836, dtype=gmm.dtype)
        gmm.mu.assign(tf.reshape(tf.stack([u1, u2]), gmm.mu.get_shape()))
        gmm.sigma.assign(tf.reshape(tf.stack([o1, o2]), gmm.sigma.get_shape()))
        gmm.rou[0, 0, :, 0].assign(p)
        gmm.w[0, 0, :, 0].assign(w)
        _, _, _, lpe, _ = gmm(None, full_out=False)
        he = -tf.reduce_sum(lpe * gmm.we * tf.nn.softmax(gmm.w, axis=2))
        tolerance = 1e-5
        self.assertAllClose(e_entropy, he, rtol=tolerance, atol=tolerance)
        print(f'edge entropy from matlab: {e_entropy}')
        print(f'edge entropy from tensorflow: {he}')

    def testSanity1D(self):
        gmm = GMMComplete(c=1, n=1, l=3, k=17, dtype=tf.float64)
        mu = tf.reshape(tf.constant([-5.95, 0.95, 7.23], dtype=gmm.dtype), gmm.mu.get_shape())
        sigma = tf.reshape(tf.constant([0.75, 1.39, 0.095], dtype=gmm.dtype), gmm.sigma.get_shape())
        n_entropy = tf.constant(2.022559320969668, dtype=gmm.dtype)
        gmm.mu.assign(mu)
        gmm.sigma.assign(sigma)
        gmm.w[0, 0, :, 0].assign(tf.constant([3, 4, 3], dtype=gmm.dtype))
        _, _, lpn, _, _ = gmm(None, full_out=False)
        hn = -tf.reduce_sum(lpn * gmm.wn * tf.nn.softmax(gmm.w, axis=2))
        tolerance = 1e-2
        self.assertAllClose(n_entropy, hn, rtol=tolerance, atol=tolerance)
        print(f'node entropy from matlab: {n_entropy}')
        print(f'node entropy from tensorflow: {hn}')


if __name__ == '__main__':
    tf.test.main()  # run all unit tests
