import tensorflow as tf
from core.marginal.gmm_grid import GMMGrid
from core.mrf.grid import GridMrf
from core.potential.dnn import NeuralNetPotential
from core.potential.polynominal import PolynomialPotential


class MrfUnitTest(tf.test.TestCase):
    pass


if __name__ == '__main__':
    tf.test.main()  # run all unit tests
