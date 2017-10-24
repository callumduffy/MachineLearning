import pandas as pd
import matplotlib.pyplot as plt
import time
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
models.append(('CART', DecisionTreeClassifier()))
models.append(('SGDC', SGDClassifier()))

scoring = ['accuracy', 'homogeneity_score', 'mutual_info_score', 'completeness_score', 'v_measure_score']

size = 100

for name, model in models:

    kfold = model_selection.KFold(n_splits=10)
    start = time.time()
    cv_results = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring[0])
    cv_results1 = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring[1])
    cv_results2 = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring[2])
    cv_results3 = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring[3])
    cv_results4 = model_selection.cross_val_score(model, X[:size], Y[:size], cv=kfold, scoring=scoring[4])
    end = time.time()

    msg = "%s: \n%s: \n%s: %f (%f) " % (name, "5 Common Evaluation Metrics ", "Accuracy" ,  cv_results.mean(), cv_results.std())
    msg1 = "%s: %f (%f) " % ("homogeneity_score" ,  cv_results1.mean(), cv_results1.std())
    msg2 = "%s: %f (%f) " % ("mutual_info_score" ,  cv_results2.mean(), cv_results2.std())
    msg3 = "%s: %f (%f) " % ("completeness_score" ,  cv_results3.mean(), cv_results3.std())
    msg4 = "%s: %f (%f) " % ("v_measure_score" ,  cv_results4.mean(), cv_results4.std())
    msg5 = "%s: \n%s: %f" % (" 2 New Evaluation Metrics", "Times (seconds)" , (end-start) )
    #msg5 = "%s: %f" % ("Times (seconds)" , (end-start) )
    print(msg, "\n",msg1 ,"\n",msg2,"\n",msg3,"\n",msg4,"\n",msg5, "\n" )



    
