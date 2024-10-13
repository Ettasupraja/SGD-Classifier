# SGD-Classifier

NAME: ETTA SUPRAJA

REG NO: 212223220022
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import Necessary Libraries and Load Data
2.Split Dataset into Training and Testing Sets
3.Train the Model Using Stochastic Gradient Descent (SGD)
4.Make Predictions and Evaluate Accuracy
5.Generate Confusion Matrix
```
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: ETTA SUPRAJA
RegisterNumber:  212223220022
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

Output:

![375938702-e67c9279-7fa9-412a-ad31-76ad207dfcdc](https://github.com/user-attachments/assets/ce123014-6cf9-4a1d-85c9-03b531be2cf5)

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

Output:
![374886907-a865fcc4-6fa3-447f-a802-4d9ee8413bf5](https://github.com/user-attachments/assets/d75e92d3-b2eb-40ca-8bf8-e9254b40472c)

y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

Output:

![374887523-bc2515cd-88eb-438e-8a84-5adc5960e686](https://github.com/user-attachments/assets/ad7ab23f-f887-423e-972e-c3d2d1a279f9)


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

Output:
![375938731-45eddd4e-b547-4e35-b63f-1983c8248598](https://github.com/user-attachments/assets/16871da1-0a6f-4e36-b9ce-c3d44c444aae)

```


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
