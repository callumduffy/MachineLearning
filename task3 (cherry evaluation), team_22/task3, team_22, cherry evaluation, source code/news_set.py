import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# load dataset
dataset = pd.read_csv("OnlineNewsPopularity.csv", delimiter=", ").drop(['url'], axis=1)

X = dataset
Y = dataset['shares']

X= X.astype('int')
Y=Y.astype('int')

# prepare configuration for cross validation test harness
# prepare models, one novel and 3 baseline

models = []
#novel below
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('SGDC', SGDClassifier()))

# evaluate each model in turn
scoring = ['accuracy', 'homogeneity_score', 'normalized_mutual_info_score', 'completeness_score', 'v_measure_score']
size = 1000

for name, model in models:
    print("\n%s" %(name))

    for score in scoring:
        kfold = model_selection.KFold(n_splits=10)
        #set to check time for accuracy of each algo
        if score == 'accuracy':
            start = time.time()
            cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=score)
            end = time.time()
            runtime = end-start
            timescore = cv_results.mean()/runtime
            msg1 = "%s: %f (%f) Timescore: %f " % (score ,  cv_results.mean(), cv_results.std(),timescore)
        else:
            cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=score)
            msg1 = "%s: %f (%f) " % (score ,  cv_results.mean(), cv_results.std())
        print(msg1)
