import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
dataset = pd.read_csv("Twitter.csv", delimiter=",")
dataset.drop(dataset.columns[list(range(7,77))], axis=1, inplace=True)

dataset.columns=['NCD_0', 'NCD_1','NCD_2','NCD_3','NCD_4','NCD_5','NCD_6', 'Mean']
sizes = [100, 500, 1000, 5000,10000]
X = dataset
Y = dataset['Mean']

X= X.astype('int')
Y=Y.astype('int')
# prepare configuration for cross validation test harness
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
score2 = 'neg_mean_squared_error'

for size in sizes:
    print("\nSize is %d" % (size))
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=10)
    	cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=score2)

    	msg = "%s: %f (%f)" % (name, np.sqrt(cv_results.mean()*-1), cv_results.std())
    	print(msg)
