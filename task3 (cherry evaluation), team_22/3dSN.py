import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# load dataset
dataset = pd.read_csv("OnlineNewsPopularity.csv", delimiter=", ").drop(['url'], axis=1)

X = dataset
Y = dataset['']
X= X.astype('int')
Y=Y.astype('int')

# prepare configuration for cross validation test harness
# prepare models, one novel and 3 baseline

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('GP', GaussianProcessClassifier()))
models.append(('SGD', SGDClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'popularity'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
