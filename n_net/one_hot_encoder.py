import numpy as np
import sys

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from n_net.nnCostFunction import nnCostFunction, perceptronCostFunction
from n_net.predict import predict, encode_and_scale
from n_net.preprocess import preprocess


def initialize_weights(input_units, hidden_units):
    if hidden_units == 0:
        Theta1 = np.random.uniform(-0.01, 0.01, (1, input_units + 1))
    else:
        Theta1 = np.random.uniform(-0.01, 0.01, (hidden_units, input_units + 1))
    Theta2 = np.random.uniform(-0.01, 0.01, (1, hidden_units + 1))
    return Theta1, Theta2


def print_matches(activation, predicted_class, test_y, class_dict):
    length = len(test_y)
    actual_predicted = zip(predicted_class, test_y)
    for idx, tuple in enumerate(actual_predicted):
        print(
            "Activation {0}\t\t\t\tPredicted class : {1}\t\tActual class : {2}".format(activation[idx],
                                                                                       class_dict[tuple[0]],
                                                                                       class_dict[tuple[1]]))

    accuracy = accuracy_score(predicted_class, test_y)
    print(
        "Number of correctly classisfied instances {0} : Number of Incorrectly classisfied instances {1}"
            .format(round(accuracy * length), length - round(accuracy * length)))


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
    input_bias = np.random.uniform(-0.01, 0.01, (1, 1))
    hidden_bias = np.random.uniform(-0.01, 0.01, (1, 1))

    train_X, train_y = shuffle(train_X, train_y)
    J = 0
    for x in xrange(epoch):
        cross_entropy = 0
        for index in xrange(training_data_size):
            cross_entropy += J
            if hidden_units == 0:
                [J, Theta1_grad] = perceptronCostFunction(Theta1, train_X[index:index + 1], train_y[index])
                Theta1 -= (learning_rate * Theta1_grad)
                cross_entropy += J
            else:
                [J, Theta1_grad, Theta2_grad] = nnCostFunction(Theta1, Theta2, train_X[index:index + 1], train_y[index],
                                                               input_bias, hidden_bias)
                Theta1 -= (learning_rate * Theta1_grad)
                Theta2 -= (learning_rate * Theta2_grad)
                cross_entropy += J
        if hidden_units == 0:
            Theta = [Theta1]
        else:
            Theta = [Theta1, Theta2]

        activation, predicted_class = predict(Theta, train_X, input_bias, hidden_bias)
        accuracy = accuracy_score(predicted_class, train_y)
        print(
            "{0}\t{1}\t{2}\t{3}"
                .format(x, cross_entropy, round(accuracy * training_data_size),
                        training_data_size - round(accuracy * training_data_size)))

    activation, predicted_class = predict(Theta, test_X, input_bias, hidden_bias)
    print_matches(activation, predicted_class, test_y, class_dict)
