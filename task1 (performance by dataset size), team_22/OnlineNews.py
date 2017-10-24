import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# load dataset
dataset = pd.read_csv("OnlineNewsPopularity.csv", delimiter=",")

toDrop = list(range(0,39))
toDrop = toDrop+ (list(range(45,59)))

dataset.drop(dataset.columns[toDrop], axis=1, inplace=True)

sizes = [100, 500, 1000, 5000,10000]
X = dataset
Y = dataset[' shares']
print(X)

print(Y)
X= X.astype('int')
Y=Y.astype('int')
# prepare configuration for cross validation test harness
# prepare models
models = []
models.append(('LogisticRegressionR', LogisticRegression(), 0))
models.append(('KNN', KNeighborsClassifier(),0))
models.append(('Linear Regression', LinearRegression(),1))
models.append(('Ridge Regression', Ridge(alpha=0.5),1))
# evaluate each model in turn
scoring = ['accuracy','neg_mean_squared_error']
for size in sizes:
    print("\nSize is %d" % (size))
    for name, model, score in models:
    	kfold = model_selection.KFold(n_splits=10)
    	cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring[score])
    	msg = "%s: %f (%f) using %s for score" % (name, cv_results.mean(), cv_results.std(),  scoring[score])
    	print(msg)
