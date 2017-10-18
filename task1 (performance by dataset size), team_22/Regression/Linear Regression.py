import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("The SUM dataset, with noise.csv", delimiter=";").drop(['Instance','Feature 5 (meaningless)'], axis=1)

features = dataset.drop(['Noisy Target', 'Noisy Target Class'], axis=1)
labels = dataset['Noisy Target']

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
