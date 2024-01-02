import pandas as pd
import numpy as np
import os as os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# KNN Prediction

dataset = pd.read_csv('"E:\Machine Learning\KNN Algorithms\diabetes.csv"')
print(len(dataset))
print(dataset.head())
# replace zeroes
zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for column in zero_not_accepted:
  dataset[column] = dataset[column].replace(0,np.NaN)
  mean = int(dataset[column].mean(skipna = True))
  dataset[column] = dataset[column].replace(np.NaN,mean)

print(dataset['Glucose'])

# train and test data / split data scaling
X = dataset.iloc[:,0:8] # : means all rows and the number means column number
y = dataset.iloc[:,8]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0,test_size = 0.2)

#feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#then define the model using knn classifier and fit the train data in the model
# define the model : Init K-NN
classifier = KNeighborsClassifier(n_neighbors = 11, p = 2, metric='euclidean')
#fit model
classifier.fit(X_train, y_train)
KNeighborsClassifier(algorithm = 'auto',leaf_size = 30, metric = 'euclidean',metric_params=None,n_jobs=1 , n_neighbors = 11, p = 2, weights='uniform')
import math
y_root = math.sqrt(len(y_test))
print(y_root)
#predict the test set results
y_pred = classifier.predict(X_test)
y_pred
#it's important to evaluate the model , let's use confustion matrix to do that:

cm = confusion_matrix(y_test,y_pred)
print(cm)

#printing f1_score 
print("The f1_score is : ",f1_score(y_test,y_pred))
#printing accuracy of model
print("Accuracy of Model",accuracy_score(y_test,y_pred))
