
#Titanic Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Importing the dataset
data = pd.read_csv('C:/Users/KAMESH/Desktop/Shroov/Practice_Prgms/Titanic/train.csv')
col_target = ['Survived']
col_train = ['Age','Pclass','Sex','Fare']
X = data[col_train]
Y = data[col_target]

X['Age']=X['Age'].fillna(X['Age'].median())
X['Age'].isnull().sum()

d = {'male':0,'female':1}
X['Sex'] = X['Sex'].apply(lambda x:(d[x]))
X['Sex'].head()



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X = sc.fit_transform(X)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X, Y)

# Predicting the Test set results
Y_pred = classifier.predict(X)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, Y_pred)
print (cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = Y, cv = 5)
accuracies.mean()




