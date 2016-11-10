from n_net.sigmoid import sigmoid
import numpy as np


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    z2 = np.dot(Theta1, X.transpose())
    a2 = sigmoid(z2)
    a2 = np.vstack((np.ones((1, m)), a2))

    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3)

    return a3

    # p = zeros(size(X, 1), 1);
    # % You need to return the following variables correctly
    #
    # num_labels = size(Theta2, 1);
    # m = size(X, 1);
