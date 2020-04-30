import datetime
import numpy as np
import tensorflow as tf
from setup import select_device
from mrf2 import CompleteMrf

if __name__ == '__main__':
    # only using one GPU
    select_device(2)
    dtype = tf.float64
    datasets = {'samples2d': {'n': 2, 'c': 1, 'size': 2000, 'vs': 0}}
    name = 'samples2d'
    c = datasets[name]['c']
    n = datasets[name]['n']
    data_size = datasets[name]['size']
    vs = datasets[name]['vs']
    ds = tf.data.experimental.CsvDataset(f'./{name}.data', tf.zeros([n], dtype)).map(lambda *x: tf.stack(x))
    data = tf.stack([i for i in ds])  # shape = [total_num_of_example, features]
    """Feature standardization"""
    # mean, variance = tf.nn.moments(data, axes=[0])
    # data = tf.nn.batch_normalization(data, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-9)

    data = tf.transpose(tf.reshape(data, [c, -1, n]), [1, 0, 2])  # shape=[size, class, node]
    batch_size = data_size
    infer_steps = 2
    ir = 0.05
    tr = 0.00007
    L = 2
    T = 1
    potential = 'nn'  # 'nn'=neural_nets, 'ply'=polynomial

    entropy = 2.437028185188807  # this was calculated by matlab
    mu = np.array([[-1.0, 1.0], [1.0, -1.0]])
    o1, o2, p = 1.0, 0.5, -0.54
    o1_, o2_, p_ = 0.5, 0.8, 0.38
    sigma = np.array([[o1, o2], [o1_, o2_]])
    rou = np.array([p, p_])

    mrf = CompleteMrf(num_class=c, node=n, points=99, node_units=(7, 7, 7, 7), edge_units=(12, 12, 12, 12), t=T,
                      mvn={'mu': mu, 'sigma': sigma, 'rou': rou},
                      mw=tf.constant([0.3, 0.7], dtype=dtype), ent=entropy,
                      train_rate=tr, infer_rate=ir, mixture=L, potential=potential, dtype=dtype)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = './tb2/' + time + \
              f'-{name}-{potential}-infer{infer_steps}_ir{ir}-bs{batch_size}_tr{tr}-m{L}(test-pt)_t{T}_u10-q99'
    mrf.train(data, valid_size=vs, infer_iter=infer_steps, batch_size=batch_size, epochs=10000, log_dir=log_dir)

    # # plot the potential function and compare with the initial one
    # x = tf.cast(tf.linspace(-30., 30., 3000), tf.float64)
    # xe1, xe2 = tf.meshgrid(x, x)
    # xe = tf.reshape(tf.concat([tf.reshape(xe1, [-1, 1]), tf.reshape(xe2, [-1, 1])], -1), [1, 1, 1, -1, 2])
    # xn = tf.tile(tf.reshape(x, [1, 1, 1, -1]), [1, 2, 1, 1])
    # fn, fe = mrf.potential((xn, xe))
    # import numpy as np
    # import scipy.io
    # scipy.io.savemat('pot-2.mat', dict(fn=fn.numpy(), fe=fe.numpy()))
    #
    # for i in range(len(mrf.pot)):
    #     tmp = mrf.potential.trainable_variables[i].read_value().numpy()
    #     mrf.potential.trainable_variables[i].assign(mrf.pot[i])
    #     mrf.pot[i] = tmp
    # fn0, fe0 = mrf.potential((xn, xe))
    # scipy.io.savemat('pot0-2.mat', dict(fn=fn0.numpy(), fe=fe0.numpy()))
