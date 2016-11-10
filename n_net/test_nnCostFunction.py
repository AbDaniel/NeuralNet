import numpy as np
from unittest import TestCase

from n_net.nnCostFunction import nnCostFunction
from n_net.predict import predict


class TestNnCostFunction(TestCase):
    def test_nnCostFunction(self):
        input_layer_size = 2
        hidden_layer_size = 4
        X = np.array([[0.1683, -0.1923],
                      [0.1819, -0.1502],
                      [0.0282, 0.0300],
                      [-0.1514, 0.1826],
                      [-0.1918, 0.1673],
                      [-0.0559, -0.0018],
                      [0.1314, -0.1692],
                      [0.1979, -0.1811],
                      [0.0824, -0.0265],
                      [-0.1088, 0.1525],
                      [-0.2000, 0.1913],
                      [-0.1073, 0.0542],
                      [0.0840, -0.1327],
                      [0.1981, -0.1976],
                      [0.1301, -0.0808],
                      [-0.0576, 0.1103]])

        y = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        Theta1 = np.array([[0.8415, 0.4121, -0.9614],
                           [0.1411, -1.0000, 0.1499],
                           [-0.9589, 0.4202, 0.8367],
                           [0.6570, 0.6503, -0.8462]])

        Theta2 = np.array([[0.5403   ,-0.9900,   0.2837,    0.7539,   -0.9111]])

        num_labels = 1

        [J, Theta1_grad, Theta2_grad] = nnCostFunction(Theta1, Theta2, num_labels, X, y)
        score = predict(Theta1, Theta2, X)
        print(score)
        print(J)
        # for x in xrange(50):
        #
        #     Theta1 = np.subtract(Theta1, Theta1_grad)
        #     Theta2 = np.subtract(Theta2, Theta2_grad)
        #     print("Error =" + str(J))

            # score = predict(Theta1, Theta2, X)
            # print("Hello World")
