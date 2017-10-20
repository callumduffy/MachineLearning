import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def transpose(lst):
    temp = zip(* lst)
    temp2 = [list(i) for i in temp]
    return temp2
# load dataset
dataset = pd.read_csv("The SUM dataset, with noise.csv", delimiter=";").drop(['Instance','Feature 5 (meaningless)'], axis=1)
dataset = dataset[:15]
dataset.loc[dataset['Noisy Target Class'] == "Very Large Number", 'Noisy Target Class'] = 1
dataset.loc[dataset['Noisy Target Class'] == "Large Number", 'Noisy Target Class'] = 0
dataset.loc[dataset['Noisy Target Class'] == "Medium Number", 'Noisy Target Class'] = 0
X = dataset
Y = dataset['Noisy Target Class']
X= X.astype('int')
Y=Y.astype('int')

print(X)

print(Y)
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
for name, model in models:

	kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
