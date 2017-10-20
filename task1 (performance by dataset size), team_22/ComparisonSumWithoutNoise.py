import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
dataset = pd.read_csv("The SUM dataset, without noise.csv", delimiter=";").drop(['Instance','Feature 5 (meaningless)'], axis=1)

dataset.loc[dataset['Target Class'] == "Very Large Number", 'Target Class'] = 4
dataset.loc[dataset['Target Class'] == "Large Number", 'Target Class'] = 3
dataset.loc[dataset['Target Class'] == "Medium Number", 'Target Class'] = 2
dataset.loc[dataset['Target Class'] == "Small Number", 'Target Class'] = 1
dataset.loc[dataset['Target Class'] == "Very Small Number", 'Target Class'] = 0

sizes = [100, 500, 1000, 5000,10000]

X = dataset
Y = dataset['Target Class']

X= X.astype('int')
Y=Y.astype('int')
# prepare configuration for cross validation test harness
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for size in sizes:
    print("\nSize is %d" % (size))
    for name, model in models:

    	kfold = model_selection.KFold(n_splits=10)
    	cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
