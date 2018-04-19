import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data=pd.read_csv("C:/Users/User/Downloads/train.csv").as_matrix()
clf=DecisionTreeClassifier()

xtrain=data[0:21000,1:]
train_label=data[0:21000,0]

clf.fit(xtrain,train_label)

xtest=data[21000:,1:]
actual_label=data[21000:,0]

d=xtrain[8]
d.shape=(28,28)
pt.imshow(255-d,cmap='gray')
print(clf.predict([xtrain[8]]))
pt.show()
