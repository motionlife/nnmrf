import numpy as np

# import sklearn.datasets as ds

"""
    x1
   /  \
 x2----x3
"""
# sampling from mixture of multivariate gausion

mixture_component = 2
mixture_weights = [0.3, 0.7]
sample_size = 2000
dims = 2

# o1 = sklearn.datasets.make_spd_matrix(3, random_state=5)
# o2 = sklearn.datasets.make_spd_matrix(3, random_state=9)
# o3 = sklearn.datasets.make_spd_matrix(3, random_state=10)
# mean = np.array([[-2.5, 1.7, -1], [1.9, -1.5, 2.2]])
# o1, o2, o3 = 0.5, 1.2, 0.7
# p1, p2, p3 = -0.3, 0.4, 0.7
# C = np.array([[o1 ** 2, p1 * o1 * o2, p2 * o1 * o3],
#               [p1 * o1 * o2, o2 ** 2, p3 * o2 * o3],
#               [p2 * o1 * o3, p3 * o2 * o3, o3 ** 2]])
#
# o1_, o2_, o3_ = 0.9, 0.7, 1.
# p1_, p2_, p3_ = 0.5, 0.3, -0.5
# C_ = np.array([[o1_ ** 2, p1_ * o1_ * o2_, p2_ * o1_ * o3_],
#                [p1_ * o1_ * o2_, o2_ ** 2, p3_ * o2_ * o3_],
#                [p2_ * o1_ * o3_, p3_ * o2_ * o3_, o3_ ** 2]])

# entropy =  3.454530625336702 # this was calculated by matlab

mean = np.array([[-1, 1], [1, -1]])
o1, o2, p = 1.0, 0.5, -0.54
o1_, o2_, p_ = 0.5, 0.8, 0.38
C = np.array([[o1 ** 2, p * o1 * o2],
              [p * o1 * o2, o2 ** 2]])

C_ = np.array([[o1_ ** 2, p_ * o1_ * o2_],
               [p_ * o1_ * o2_, o2_ ** 2]])

variance = np.stack([C, C_], axis=0)

samples = np.zeros(shape=[sample_size, dims])

for i in range(sample_size):
    l = np.random.choice(mixture_component, 1, p=mixture_weights)[0]
    x = np.random.multivariate_normal(mean=mean[l, :], cov=variance[l, :, :], size=1)
    samples[i, :] = x

np.savetxt("samples2d.data", X=samples, delimiter=',')

# np.all(np.linalg.eigvals(C) > 0)
# o1, o2, o3 = 0.5, 1.2, 1.7
# p1, p2, p3 = -0.3, 0.4, 0.7
# C = np.array([[o1 ** 2, p1 * o1 * o2, p2 * o1 * o3],
#               [p1 * o1 * o2, o2 ** 2, p3 * o2 * o3],
#               [p2 * o1 * o3, p3 * o2 * o3, o3 ** 2]])

# o1, o2, o3 = 0.9, 0.7, 1.9
# p1, p2, p3 = 0.5, 0.3, -0.2
#
# o1, o2, o3 = 1.17, 2.25, 0.98
# p1, p2, p3 = 0.25, -0.77, 0.38
