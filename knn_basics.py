# https://www.kaggle.com/wenruliu/adult-income-dataset

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# data
data=pd.read_csv(r'/home/ram/Downloads/kaggle/adult.csv')
print(data.info())
print(data.isnull().sum())
l=list(data.columns)
for i in range(len(l)):
    print(data[l[i]].value_counts())


data = data[data["workclass"] != "?"]
data = data[data["occupation"] != "?"]
data = data[data["native-country"] != "?"]

data["income"]=[1 if n==">50K" else 0 for n in data["income"]]

#print(y.value_counts())

#The field marital-status are too detailed.
# It can be re-categorizing into more general terms. Discretisation will be applied on this field later.
# Now how about the target value?

# We are faced with a classification problem on two classes.

# Are the two classes balanced?

print(data["income"].value_counts()[0] / data.shape[0])
print(data["income"].value_counts()[1] / data.shape[0])

# The two classes are imbalanced.

# Stratified sampling will be adopted in dividing train and test set to preserve the ratio between two classes



# Deal with categorical columns
# To fit the data into prediction model, we need convert categorical values to numerical ones.
#
# Before that, we will evaluate if any transformation on categorical columns are necessary.
#
# Discretisation is a common way to make categorical data more tidy and meaningful.
#
# Here we apply discretisation on column marital_status

data.replace(['Divorced', 'Married-AF-spouse',
              'Married-civ-spouse', 'Married-spouse-absent',
              'Never-married','Separated','Widowed'],
             ['not married','married','married','married',
              'not married','not married','not married'], inplace = True)

# Now we can convert categorical columns to numerical representations.

for col in data.columns:
    b, c = np.unique(data[col], return_inverse=True)
    data[col] = c

print(data.head())
y=data["income"]
x=data.drop(columns="income")

x_train,x_test,y_trin,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

knn=KNeighborsClassifier()
knn.fit(x_train,y_trin)
y_pre=knn.predict(x_test)
print("accuracy:{}".format(accuracy_score(y_test,y_pre)))
print(confusion_matrix(y_test,y_test))
print(classification_report(y_test,y_pre))

