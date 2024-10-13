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

![Screenshot 2024-10-13 094237](https://github.com/user-attachments/assets/e750d063-ee75-4ad6-9d94-efa76cc3217e)


X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

Output:
![Screenshot 2024-10-13 094243](https://github.com/user-attachments/assets/06e076a5-3aa5-483c-bc55-2cdd6d4341aa)


y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

Output:

![Screenshot 2024-10-13 094247](https://github.com/user-attachments/assets/89ab4f2e-d1c5-4783-a6a4-68d5ca698ed2)



cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

Output:
![Screenshot 2024-10-13 094251](https://github.com/user-attachments/assets/ac4723e1-8ea6-45b2-976c-8c30a89a78a9)

*/
```


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
