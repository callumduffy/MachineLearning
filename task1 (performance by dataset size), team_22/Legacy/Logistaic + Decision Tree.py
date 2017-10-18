import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import csv

dataset = pd.read_csv("The SUM dataset, without noise.csv", delimiter=";").drop(['Instance','Feature 5 (meaningless)'], axis=1)


print("loaded data")

kf = KFold(n_splits=10, shuffle=True)
logistic_regression = LogisticRegression()
decision_tree = DecisionTreeClassifier()
lr_accuracys = []
dt_accuracys = []
sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, len(dataset)]

for size in sizes:
    print("\n")
    for train_indices, test_indices in kf.split(dataset[:size]):
        print("hello")
        training_set = dataset.iloc[train_indices]
        test_set = dataset.iloc[test_indices]

        X = training_set.drop(['Target', 'Target Class'], axis=1)
        Y = training_set['Target Class']

        features_test = test_set.drop(['Target', 'Target Class'], axis=1)
        labels_test = test_set['Target Class']

        logistic_regression.fit(X, Y)
        decision_tree.fit(X, Y)

        lreg_predictions = logistic_regression.predict(features_test)
        dtree_predictions = decision_tree.predict(features_test)

        lreg_accuracy = accuracy_score(labels_test, lreg_predictions)
        dtree_accuracy = accuracy_score(labels_test, dtree_predictions)

        lr_accuracys.append(lreg_accuracy)
        dt_accuracys.append(dtree_accuracy)

    print("Logistic Regression Sample Size: " + str(size) + ", Mean Value: "+ str(np.array(lr_accuracys).mean()))
    print(sum(lr_accuracys)/len(lr_accuracys))

    print("Decision Tree Sample Size: " + str(size) + ", Mean Value: "+ str(np.array(dt_accuracys).mean()))
    print(sum(dt_accuracys)/len(dt_accuracys))
