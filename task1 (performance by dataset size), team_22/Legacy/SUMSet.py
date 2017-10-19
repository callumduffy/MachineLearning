from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold # import KFold


import pandas as  pd
import numpy as np

dataset = pd.read_csv("The SUM dataset, without noise.csv", delimiter=";").drop(['Instance','Feature 5 (meaningless)'], axis=1)


def DecisionTree(dataset):                                      #Classification

    sizes =  [10,20,50,100,200, 500, 1000, 5000, 10000, 50000, 100000, 500000]

    kf = KFold(n_splits=10, shuffle =True)
    dcf = DecisionTreeClassifier()
    for size in sizes:
        dcg=[]
        for train_indices, test_indices in kf.split(dataset[:size]):
            training_set = dataset[:size]
            test_set = dataset.iloc[test_indices]
            X = training_set.drop(['Target','Target Class'], axis=1)
            Y = training_set['Target']
            features_test = test_set.drop(['Target','Target Class'], axis=1)
            labels_test = test_set['Target']
            dcf.fit(X,Y)
            predictions = dcf.predict(features_test)
            hamming = hamming_loss(labels_test, predictions)
            accuracy = accuracy_score(labels_test, predictions)
            dcg.append(accuracy)
        print("Size %d has an Accuracy of %f " % (size, sum(dcg)/len(dcg)))




def logistRegression(dataset):                      #Classification


    sizes =  [10,100,200,500, 1000, 5000, 10000, 50000, 100000, 500000]

    kf = KFold(n_splits=10, shuffle =True)

    logRe = LogisticRegression()
    lr=[]
    for size in sizes:

        for train_indices, test_indices in kf.split(dataset[:size]):
            training_set = dataset[:size]
            test_set = dataset.iloc[test_indices]
            X = training_set.drop(['Target','Target Class'], axis=1)
            Y = training_set['Target']
            features_test = test_set.drop(['Target','Target Class'], axis=1)
            labels_test = test_set['Target']
            logRe.fit(X, Y)
            predictions = logRe.predict(features_test)
            hamming = hamming_loss(labels_test, predictions)
            accuracy = accuracy_score(labels_test, predictions)
            lr.append(accuracy)

        print("Size %d has an Accuracy of %f " % (size, sum(lr)/len(lr)))
        #print("Size %d has an Hamming Loss of %f " % (size, hamming))


def LinearRe(dataset):                              #Regression

    sizes =  [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    kf = KFold(n_splits=10)
    lg = LinearRegression(1e5)
    lin=[]
    for size in sizes:
        for train_indices, test_indices in kf.split(dataset[:size]):
            training_set = dataset[:size]
            test_set = dataset.iloc[test_indices]
            X = training_set.drop(['Target','Target Class'], axis=1)
            Y = training_set['Target']
            features_test = test_set.drop(['Target','Target Class'], axis=1)
            labels_test = test_set['Target']
            lg.fit(X,Y)
            predictions = lg.predict(features_test)
            mean_err = mean_squared_error(labels_test, predictions)
            print(predictions[0])
            lin.append(mean_err)
        print("Size %d has a Root mean Error of %f " % (size, sum(lin)/len(lin)))

#DecisionTree(dataset)
LinearRe(dataset)
#logistRegression(dataset)
