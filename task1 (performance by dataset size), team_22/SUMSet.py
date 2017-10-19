from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
import pandas as  pd
import numpy as np

dataset = pd.read_csv("The SUM dataset, without noise.csv", delimiter=";").drop(['Instance','Feature 5 (meaningless)'], axis=1)


def DecisionTree(dataset):
    features = dataset.drop(['Target','Target Class'], axis=1)
    labels = dataset['Target']

    sizes =  [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]


    test_size = int(dataset.shape[0]*0.9)
    features_t = features[:-test_size]
    labels_t = labels[:-test_size]

    features_test= features[-test_size:]
    labels_test = labels[-test_size:]

    dcf = DecisionTreeClassifier()
    for size in sizes:
        dcf.fit(features_t[:size], labels_t[:size])

        predictions = dcf.predict(features_test)
        mean_err = mean_squared_error(labels_test, predictions)
        rmse = np.sqrt(mean_err)
        print("Size %d has a Root mean Error of %f " % (size, rmse))

def LinearRe(dataset):
    features = dataset.drop(['Target','Target Class'], axis=1)
    labels = dataset['Target']

    sizes =  [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]


    test_size = int(dataset.shape[0]*0.9)

    # Split the data into training/testing sets

    features_t = features[:-test_size]
    labels_t = labels[:-test_size]

    features_test= features[-test_size:]
    labels_test = labels[-test_size:]

    lg = LinearRegression()

    for size in sizes:
        lg.fit(features_t[:size], labels_t[:size])

        predictions = lg.predict(features_test)
        mean_err = mean_squared_error(labels_test, predictions)
        rmse = np.sqrt(mean_err)
        print("Size %d has a Root mean Error of %f " % (size, rmse))

DecisionTree(dataset)
LinearRe(dataset)
