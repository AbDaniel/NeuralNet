import numpy as np
import pandas as pd
import scipy.io.arff as arff
import sys
import time
from sklearn import preprocessing
from n_net.nnCostFunction import nnCostFunction
from n_net.predict import predict
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def is_classification(metadata, feature):
    return metadata[feature][0] == 'nominal'


def one_hot_encoding(df, metadata, features):
    for feature in features:
        if is_classification(metadata, feature) and not feature == 'class':
            encoded = pd.get_dummies(df[feature])
            df = df.drop(feature, axis=1)
            df = pd.concat([df, encoded], axis=1)
    return df


def preprocess(input_file):
    loaded_arff = arff.loadarff(open(input_file, 'rb'))
    (df, metadata) = loaded_arff
    features = metadata.names()

    df = pd.DataFrame(df)
    y = df['class']
    y = y.apply(lambda x: 0 if (x == metadata['class'][1][0]) else 1)
    X = df.drop('class', axis=1)

    return X, y.values, len(X), features, metadata


def encode_and_scale(train_X, test_X):
    complete_df = pd.concat([train_X, test_X])
    encoded_df = one_hot_encoding(complete_df, metadata, features)

    train_X = encoded_df[:training_data_size]
    test_X = encoded_df[training_data_size:len(encoded_df)]

    return preprocessing.scale(train_X.values), preprocessing.scale(test_X.values)


def initialize_weights(input_units, hidden_units):
    Theta1 = np.random.uniform(-0.01, 0.01, (hidden_units, input_units + 1))
    Theta2 = np.random.uniform(-0.01, 0.01, (1, hidden_units + 1))
    return Theta1, Theta2


if __name__ == '__main__':
    start_time = time.time()

    train_X, train_y, training_data_size, features, metadata = preprocess(sys.argv[1])
    test_X, test_y, test_data_size, _, _ = preprocess(sys.argv[2])
    learning_rate = float(sys.argv[3])
    hidden_units = int(sys.argv[4])
    epoch = int(sys.argv[5])

    train_X, test_X = encode_and_scale(train_X, test_X)

    input_units = train_X.shape[1]
    Theta1, Theta2 = initialize_weights(input_units, hidden_units)

    train_X, train_y = shuffle(train_X, train_y)
    for x in xrange(epoch):
        for index in xrange(training_data_size):
            [J, Theta1_grad, Theta2_grad] = nnCostFunction(Theta1, Theta2, train_X[index:index + 1], train_y[index])
            Theta1 -= (learning_rate * Theta1_grad)
            Theta2 -= (learning_rate * Theta2_grad)

        print("Epoch {0} - Cross Entropy Error: {1}".format(x, J))

    print(accuracy_score(predict(Theta1, Theta2, test_X), test_y))
    print("Hello")
