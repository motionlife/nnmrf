import numpy as np
import tensorflow as tf

# this file record the code execute after run file to sample data from distribution
# test data integrity
std_dev = tf.sqrt(variance)
# original = data * std_dev + mean
marginal = mrf.marginal
mu = marginal.mu
sigma = marginal.sigma
rou = marginal.rou
alpha = tf.nn.softmax(marginal.w, axis=2)

sample_size_each_class = 70
samples = ""
classes = ["Kama", "Rosa", "Canadian"]
shorts = ["k", "r", "c"]
for i in tf.range(c):
    # samples.append(f"_________class:{i}___________")
    a = alpha[i, 0, :, 0]
    seg = [1] * L
    seg[0] = a[0]
    # todo use np.random.choice()
    for j in range(1, L - 1):
        seg[j] = seg[j - 1] + a[j]
    for s in range(sample_size_each_class):
        rd = np.random.uniform()
        l = k = 0
        for k in range(1, L):
            if seg[k - 1] <= rd < seg[k]: l = k
        u = mu[i, :, l, 0]
        o = sigma[i, :, l, 0]
        x = np.random.multivariate_normal(u, np.diag(o), 1)
        x = x * std_dev + mean
        # output format for t-SNE visualization
        item = f',["MRF-{shorts[i]}{s}", "{classes[i]}-MRF",'
        for d in x[0, :]:
            item += str(d.numpy()) + ","
        samples += item + "]\n"

with open('samples.txt', 'a') as f:
    f.write(samples + '\n')

ss=100
samples = np.zeros([10, ss, 784])
for i in range(10):
    a = alpha[i, 0, 0, 0, :, 0]
    for j in range(ss):
        l = np.random.choice(7, 1, p=a)[0]
        u = tf.reshape(mu[i, :, :, :, l, 0], [784, ])
        o = tf.reshape(sigma[i, :, :, :, l, 0], [784, ])
        samples[i, j, :] = np.random.multivariate_normal(u, np.diag(o), 1)
