import pandas as pd
from scipy.io import arff


def preprocess(input_file):
    loaded_arff = arff.loadarff(open(input_file, 'rb'))
    (df, metadata) = loaded_arff
    features = metadata.names()

    df = pd.DataFrame(df)
    y = df['class']
    y = y.apply(lambda x: 0 if (x == metadata['class'][1][0]) else 1)
    X = df.drop('class', axis=1)

    return X, y.values, len(X), features, metadata
