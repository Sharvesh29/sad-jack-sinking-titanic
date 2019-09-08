
#Titanic Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Importing the dataset
data = pd.read_csv('C:/Users/vignesh/Desktop/Data science umps/Titanic data set/train.csv')
col_target = ['Survived']
col_train = ['Age','Pclass','Sex','Fare']
X = data[col_train]
Y = data[col_target]

X['Age']=X['Age'].fillna(X['Age'].median())
X['Age'].isnull().sum()

d = {'male':0,'female':1}
X['Sex'] = X['Sex'].apply(lambda x:(d[x]))
X['Sex'].head()

X.head(60)
# Splitting the dataset into the Training set and Test set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print (cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)
accuracies.mean()
accuracies.std()




