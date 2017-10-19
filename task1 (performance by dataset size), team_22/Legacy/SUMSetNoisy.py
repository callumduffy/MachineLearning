from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold # import KFold
import matplotlib.pyplot as plt
import pandas as  pd
import numpy as np



dataset = pd.read_csv("The SUM dataset, with noise.csv", delimiter=";").drop(['Instance','Feature 5 (meaningless)'], axis=1)

def decTree(dataset):
    features = dataset.drop(['Noisy Target', 'Noisy Target Class'], axis=1)
    labels = dataset['Noisy Target']

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
        hamming = hamming_loss(labels_test, predictions)
        accuracy = accuracy_score(labels_test, predictions)
        print("Size %d has an Accuracy of %f " % (size, accuracy))
        print("Size %d has an Hamming Loss of %f " % (size, hamming))

def logistRegression(dataset):                      #Classification


    sizes =  [100,500, 1000, 5000, 10000, 50000, 100000, 500000]

    kf = KFold(n_splits=10, shuffle =True)

    logRe = LogisticRegression()
    lr=[]
    for size in sizes:

        for train_indices, test_indices in kf.split(dataset[:size]):
            #print(test_indices)
            training_set = dataset[:size]
            test_set = dataset.iloc[test_indices]
            X = training_set.drop(['Noisy Target', 'Noisy Target Class'], axis=1)
            Y = training_set['Noisy Target']
            features_test = test_set.drop(['Noisy Target', 'Noisy Target Class'], axis=1)
            labels_test = test_set['Noisy Target']
            logRe.fit(X, Y)
            predictions = logRe.predict(features_test)
            hamming = hamming_loss(labels_test, predictions)
            accuracy = accuracy_score(labels_test, predictions)
            lr.append(accuracy)

        print("Size %d has an Accuracy of %f " % (size, sum(lr)/size))
        #print("Size %d has an Hamming Loss of %f " % (size, hamming))




def linearRe(dataset):

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

#decTree(dataset)
#linearRe(dataset)
logistRegression(dataset)
