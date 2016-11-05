from n_net.sigmoid import sigmoid
import numpy as np


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    h1 = sigmoid(np.dot(X, Theta1.transpose()         ))
