import datetime
import numpy as np
import tensorflow as tf
from setup import select_device
from mrf import CompleteMrf

if __name__ == '__main__':
    # only using one GPU
    select_device(1)
    dtype = tf.float64
    datasets = {'sample': {'n': 3, 'c': 1, 'size': 2000, 'vs': 0},
                'sample3': {'n': 3, 'c': 1, 'size': 3000, 'vs': 0}}
    name = 'sample'
    c = datasets[name]['c']
    n = datasets[name]['n']
    data_size = datasets[name]['size']
    vs = datasets[name]['vs']
    ds = tf.data.experimental.CsvDataset(f'./{name}.data', tf.zeros([n], dtype)).map(lambda *x: tf.stack(x))
    data = tf.stack([i for i in ds])  # shape = [total_num_of_example, features]
    """Feature standardization"""
    mean, variance = tf.nn.moments(data, axes=[0])
    data = tf.nn.batch_normalization(data, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-9)

    data = tf.transpose(tf.reshape(data, [c, -1, n]), [1, 0, 2])  # shape=[size, class, node]
    batch_size = 100
    infer_steps = 200
    ir = 0.01
    tr = 0.001
    L = 7
    T = 1
    potential = 'nn'  # 'nn'=neural_nets, 'ply'=polynomial

    entropy = 3.454530625336702  # this was calculated by matlab
    mu = np.array([[-2.5, 1.7], [1.9, -1.5], [-1., 2.2]])
    o1, o2, o3 = 0.5, 1.2, 0.7
    o1_, o2_, o3_ = 0.9, 0.7, 1.
    p1, p2, p3 = -0.3, 0.4, 0.7
    p1_, p2_, p3_ = 0.5, 0.3, -0.5
    sigma = np.array([[o1, o1_], [o2, o2_], [o3, o3_]])
    rou = np.array([[p1, p1_], [p2, p2_], [p3, p3_]])
    # C1 = np.array([[o1 ** 2, p1 * o1 * o2, p2 * o1 * o3],
    #                [p1 * o1 * o2, o2 ** 2, p3 * o2 * o3],
    #                [p2 * o1 * o3, p3 * o2 * o3, o3 ** 2]])
    # C2 = np.array([[o1_ ** 2, p1_ * o1_ * o2_, p2_ * o1_ * o3_],
    #                [p1_ * o1_ * o2_, o2_ ** 2, p3_ * o2_ * o3_],
    #                [p2_ * o1_ * o3_, p3_ * o2_ * o3_, o3_ ** 2]])

    mrf = CompleteMrf(num_class=c, node=n, points=17, node_units=(3, 4, 4, 3), edge_units=(4, 5, 5, 4), t=T,
                      mvn=tf.transpose(tf.stack([mu, sigma, rou], 0), [2, 0, 1]),
                      mw=tf.constant([0.3, 0.7], dtype=dtype), ent=entropy,
                      train_rate=tr, infer_rate=ir, mixture=L, potential=potential, dtype=dtype)

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = './tb/' + time + \
              f'-{name}-{potential}-infer{infer_steps}_ir{ir}-bs{batch_size}_tr{tr}-m{L}(o0.001-0.5-50)_t{T}_u7-100'
    mrf.train(data, valid_size=vs, infer_iter=infer_steps, batch_size=batch_size, epochs=50, log_dir=log_dir)
