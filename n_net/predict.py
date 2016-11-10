from n_net.sigmoid import sigmoid
import numpy as np


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    z2 = np.dot(X, Theta1.transpose())
    a2 = sigmoid(z2)

    a2 = np.hstack((np.ones((m, 1)), a2))

    z3 = np.dot(a2, Theta2.transpose())
    a3 = sigmoid(z3).transpose()[0]

    threshold = np.vectorize(lambda x: 1 if(x > 0.5) else 0)

    return threshold(a3)