import numpy as np
import pandas as pd
import scipy.io.arff as arff
import sys
import time
from sklearn import preprocessing
from n_net.nnCostFunction import nnCostFunction
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

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

    return a3, threshold(a3)


def is_classification(metadata, feature):
    return metadata[feature][0] == 'nominal'


def one_hot_encoding(df, metadata, features):
    for feature in features:
        if is_classification(metadata, feature) and not feature == 'class':
            encoded = pd.get_dummies(df[feature])
            df = df.drop(feature, axis=1)
            df = pd.concat([df, encoded], axis=1)
    return df


def encode_and_scale(train_X, test_X, metadata, features):
    training_data_size = len(train_X)
    complete_df = pd.concat([train_X, test_X])
    encoded_df = one_hot_encoding(complete_df, metadata, features)

    train_X = encoded_df[:training_data_size]
    test_X = encoded_df[training_data_size:len(encoded_df)]

    return preprocessing.scale(train_X.values), preprocessing.scale(test_X.values)
