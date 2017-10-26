import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("fashion-mnist_train.csv", delimiter=",")

X = dataset
Y = dataset['label']

X= X.astype('int')
Y=Y.astype('int')

models = []
#novel below
models.append(('KNN', KNeighborsClassifier())) 
#baselines
models.append(('LR', LogisticRegression()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('SGDC', SGDClassifier()))

scoring = ['accuracy', 'homogeneity_score', 'precision', 'completeness_score', 'v_measure_score']

size = 1000

for name, model in models:
    print("\n%s" %(name))

    for score in scoring:
        kfold = model_selection.KFold(n_splits=10)
        start = time.time()
        cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=score)
        end = time.time()
        runtime = end-start

        msg1 = "%s: %f (%f) " % (score ,  cv_results.mean(), cv_results.std())
        print("Time (score/cpu-time): %f" %(cv_results.mean()/runtime))
        print(msg1)

    
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("fashion-mnist_train.csv", delimiter=",")

X = dataset
Y = dataset['label']

X= X.astype('int')
Y=Y.astype('int')

models = []
#novel below
models.append(('KNN', KNeighborsClassifier())) 
#baselines
models.append(('LR', LogisticRegression()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('SGDC', SGDClassifier()))

scoring = ['accuracy', 'homogeneity_score', 'mutual_info_score', 'completeness_score', 'v_measure_score']

size = 1000

for name, model in models:
    print("\n%s" %(name))

    for score in scoring:
        kfold = model_selection.KFold(n_splits=10)
        start = time.time()
        cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=score)
        end = time.time()
        runtime = end-start

        msg1 = "%s: %f (%f) " % (score ,  cv_results.mean(), cv_results.std())
        print("Time (score/cpu-time): %f" %(cv_results.mean()/runtime))
        print(msg1)

    
