import numpy as np
import pandas as pd
import scipy.io.arff as arff
import sys
import time
from sklearn import preprocessing
from n_net.nnCostFunction import nnCostFunction
from n_net.predict import predict


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
    return df, features, metadata


if __name__ == '__main__':
    start_time = time.time()

    df, features, metadata = preprocess(sys.argv[1])
    test_df, _, _ = preprocess(sys.argv[2])

    train_y = df['class']
    train_y = train_y.apply(lambda x: 0 if (x == 'negative') else 1)
    df = df.drop('class', axis=1)

    test_y = test_df['class']
    test_y = test_y.apply(lambda x: 0 if (x == 'negative') else 1)
    test_df = test_df.drop('class', axis=1)

    training_data_size = len(df)
    test_data_size = len(test_df)

    complete_df = pd.concat([df, test_df])
    encoded_df = one_hot_encoding(complete_df, metadata, features)

    train_df = encoded_df[:training_data_size]
    test_df = encoded_df[training_data_size:len(encoded_df)]

    input_units = train_df.shape[1]
    hidden_units = 4

    Theta1 = np.random.rand(hidden_units, input_units + 1)
    Theta2 = np.random.rand(1, hidden_units + 1)

    min_max_scaler = preprocessing.MinMaxScaler()
    train_df = min_max_scaler.fit_transform(train_df.values)

    J = 0
    for x in xrange(500):
        new_train_df = train_df[x:x + 1]

        # print("Error Before =" + str(J))
        [J, Theta1_grad, Theta2_grad] = nnCostFunction(Theta1, Theta2, 1, new_train_df, train_y.values[x])
        Theta1 = Theta1_grad
        Theta2 = Theta2_grad
        # print("Error After =" + str(J))
        # print(str(predict(Theta1, Theta2, train_df)))
        #
    print("Hello")
