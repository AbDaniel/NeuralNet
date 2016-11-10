import numpy as np
from sklearn import preprocessing

from n_net.sigmoid import sigmoid


def vectorize_y(y, num_labels):
    new_y = np.array([(yi - 1) for yi in y])
    b = np.zeros((new_y.size, num_labels))
    b[np.arange(new_y.size), new_y] = 1
    return b


def nnCostFunction(Theta1, Theta2, num_labels, X, y):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    z2 = np.dot(Theta1, X.transpose())
    a2 = sigmoid(z2)
    a2 = np.vstack((np.ones((1, m)), a2))

    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3)

    Y = y

    term1 = sum(np.multiply(-Y, np.log(a3)))
    term2 = sum(np.multiply((1 - Y), np.log(1 - a3)))

    J = sum(term1 - term2)/m

    delta3 = a3 - Y
    activation = np.multiply(a2, (1 - a2))
    temp = np.dot(Theta2.transpose(), delta3)
    delta2 = np.multiply(temp, activation)

    Theta2_grad = (Theta2_grad + np.dot(delta3, a2.transpose()))/m

    temp = np.dot(delta2, X)
    temp = np.delete(temp, 0, 0)

    Theta1_grad = (Theta1_grad + temp)/m
    return J, Theta1_grad, Theta2_grad
