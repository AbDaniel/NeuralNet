import numpy as np
import sys
import time

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from n_net.nnCostFunction import nnCostFunction
from n_net.predict import predict, encode_and_scale
from n_net.preprocess import preprocess


def initialize_weights(input_units, hidden_units):
    Theta1 = np.random.uniform(-0.01, 0.01, (hidden_units, input_units + 1))
    Theta2 = np.random.uniform(-0.01, 0.01, (1, hidden_units + 1))
    return Theta1, Theta2


def print_matches(activation, predicted_class, test_y, class_dict):
    actual_predicted = zip(predicted_class, test_y)
    for idx, tuple in enumerate(actual_predicted):
        print("Activation o/p:{0}\tPredicted class : {1}\tActual class : {2}".format(activation[idx], class_dict[tuple[0]], class_dict[tuple[1]]))


if __name__ == '__main__':
    train_X, train_y, training_data_size, features, metadata = preprocess(sys.argv[1])
    test_X, test_y, test_data_size, _, _ = preprocess(sys.argv[2])
    learning_rate = float(sys.argv[3])
    hidden_units = int(sys.argv[4])
    epoch = int(sys.argv[5])
    class_dict = {
        0: metadata['class'][1][0],
        1: metadata['class'][1][1],
    }

    train_X, test_X = encode_and_scale(train_X, test_X, metadata, features)

    input_units = train_X.shape[1]
    Theta1, Theta2 = initialize_weights(input_units, hidden_units)

    train_X, train_y = shuffle(train_X, train_y)
    J = 0
    for x in xrange(epoch):
        for index in xrange(training_data_size):
            [J, Theta1_grad, Theta2_grad] = nnCostFunction(Theta1, Theta2, train_X[index:index + 1], train_y[index])
            Theta1 -= (learning_rate * Theta1_grad)
            Theta2 -= (learning_rate * Theta2_grad)

        activation, predicted_class = predict(Theta1, Theta2, test_X)
        print_matches(activation, predicted_class, test_y, class_dict)
        # accuracy = accuracy_score(predicted, test_y)
        # print("{0} : CEE {1} : Correctly classisfied {2} : Incorrectly classisfied {3}"
        #       .format(x, J, round(accuracy * input_units), input_units - round(accuracy * input_units)))
