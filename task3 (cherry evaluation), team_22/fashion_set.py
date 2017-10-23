import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

dataset = pd.read_csv("fashion-mnist_train.csv", delimiter=",")

X = dataset
Y = dataset['label']

X= X.astype('int')
Y=Y.astype('int')

models = []
#novel below
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier()))
models.append(('GPC', GaussianProcessClassifier()))
models.append(('SGDC', SGDClassifier()))

results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring)
    results.append(cv_results)
   	names.append(name)
   	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   	print(msg)
