# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:

```
Program to implement the SVM For Spam Mail Detection..
Developed by: S Kantha Sishanth
RegisterNumber: 212222100020
```

```py
import chardet
file='/content/spam (1).csv'
with open(file, 'rb') as rawdata:
    print('Result output')
    result = chardet.detect(rawdata.read(10000))
    result

import pandas as pd
data=pd.read_csv("spam.csv",encoding="windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
print("y_pred")
y_pred


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("accuracy")
accuracy
```

## Output:

![ml9_1](https://github.com/Skanthasishanth/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118298456/285e6775-8a6e-4cc7-a9db-5ea1f8dad2ec)


### data.head():

![ml9_2](https://github.com/Skanthasishanth/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118298456/82fc1f31-c29f-4780-9882-7ec28be4fc79)

### data.info():

![ml9(3)](https://github.com/Skanthasishanth/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118298456/827af385-e6d5-45b6-80c5-12d9a9c0e973)

### data.isnull().sum():

![ml9(4)](https://github.com/Skanthasishanth/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118298456/c461664d-3267-4a76-9eb1-62ee2b53b086)


### Y_prediction value:

![ml9_5](https://github.com/Skanthasishanth/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118298456/7b3ba843-ff86-417b-bc98-a64d3b656ac5)

### Accuracy value:

![ml9_6](https://github.com/Skanthasishanth/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118298456/f3187852-c5d6-45da-815c-b9f3ae2e1601)


## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
