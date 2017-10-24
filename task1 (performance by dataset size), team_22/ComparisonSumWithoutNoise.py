import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# load dataset
dataset = pd.read_csv("The SUM dataset, without noise.csv", delimiter=";").drop(['Instance'], axis=1)

dataset.loc[dataset['Target Class'] == "Very Large Number", 'Target Class'] = 4
dataset.loc[dataset['Target Class'] == "Large Number", 'Target Class'] = 3
dataset.loc[dataset['Target Class'] == "Medium Number", 'Target Class'] = 2
dataset.loc[dataset['Target Class'] == "Small Number", 'Target Class'] = 1
dataset.loc[dataset['Target Class'] == "Very Small Number", 'Target Class'] = 0

sizes = [100, 500, 1000, 5000,10000, 50000,1000000]

X = dataset
Y = dataset['Target Class']

X= X.astype('int')
Y=Y.astype('int')
# prepare configuration for cross validation test harness
# prepare models
models = []
#models.append(('LogisticRegressionR', LogisticRegression(), 0))
#models.append(('KNN', KNeighborsClassifier(),0))
models.append(('Linear Regression', LinearRegression(),1))
models.append(('Ridge Regression', Ridge(),1))
# evaluate each model in turn
scoring = ['homogeneity_score','explained_variance']

for size in sizes:
    print("\nSize is %d" % (size))
    for name, model, score in models:
    	kfold = model_selection.KFold(n_splits=10)
    	cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring[score])
    	msg = "%s: %f (%f) using %s for score" % (name, (cv_results.mean()), cv_results.std(),  scoring[score])
    	print(msg)
