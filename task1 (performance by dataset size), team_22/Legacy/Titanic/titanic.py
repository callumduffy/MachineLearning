def toFloat(list):
    newList = [float(i) for i in list]
    return newList

def transpose(lst):
    temp = zip(* lst)
    temp2 = [list(i) for i in temp]
    return temp2

def sex(lst):
    for index, sex in enumerate(lst):
        print(sex)
        if sex == 'male':
            lst[index]=1
        else:
            lst[index]=0
    print(lst)
#    var = toFloat(lst)
    return lst

import array
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import csv

data =open('titanic dataset.csv')
data = csv.reader(data)
data = list(data)

columnNames= data[0]

data = transpose(data[1:])



data2 = [sex(data[columnNames.index('Sex')])]

data2 = transpose(data2)

dataSet = data[1:]


#print(dataSet)
y = dataSet[columnNames.index('Survived')]
y_min = min(y)
y_max = max(y)


lr =linear_model.LinearRegression()

#print(y)

predicted = cross_val_predict(lr, data2 , y , cv=1)
fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y_min, y_max], [y_min, y_max], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
