# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
### NAME: KEERTHANA V
### REG NO: 212223220045

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection & Preprocessing

2. Select relevant features that impact placement

3. Import the Logistic Regression model from sklearn.

4. Train the model using the training dataset.

5. Use the trained model to predict placement for new student data.
 

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KEERTHANA V
RegisterNumber: 212223220045 
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
## Output:
<img width="730" alt="image" src="https://github.com/user-attachments/assets/5e69cdf2-c158-4cd9-88f2-9853c60ea08e" />

```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
## Output:
<img width="659" alt="image" src="https://github.com/user-attachments/assets/1295906c-94f5-43e7-a0dc-80f9996b32c7" />

```
data1.isnull().sum()
```

## Output:
<img width="182" alt="image" src="https://github.com/user-attachments/assets/b1985f56-e934-4563-a3e6-4925cece8f1f" />

```
data1.duplicated().sum()
```
## Output:
<img width="27" alt="image" src="https://github.com/user-attachments/assets/6bb2a45a-96c6-4c26-962e-f991f0717e2f" />

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1
```
## Output:

<img width="643" alt="image" src="https://github.com/user-attachments/assets/defd2c45-9425-475b-9c08-70f238714e28" />

```
x=data1.iloc[:,:-1]
x
```
## Output:

<img width="610" alt="image" src="https://github.com/user-attachments/assets/a04216d3-6d6d-42ad-9e6b-9cbc3a96d3f4" />

```
y=data1["status"]
y
```

## Output:

<img width="287" alt="image" src="https://github.com/user-attachments/assets/fb40ff7c-8f06-4d52-aea0-24a242238cc1" />

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
## Output:

<img width="497" alt="image" src="https://github.com/user-attachments/assets/1d58a6b7-33e9-4116-973a-af7d7cdf4e6f" />

```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
## Output:
<img width="159" alt="image" src="https://github.com/user-attachments/assets/d96a93e9-ac50-40f6-8e2a-d1c00346b3a4" />

```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
## Output:

<img width="252" alt="image" src="https://github.com/user-attachments/assets/7209abe5-ad78-4d59-a75d-98cfe26cabc7" />

```

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:

<img width="847" alt="image" src="https://github.com/user-attachments/assets/c3420c31-15b5-4acd-8736-4bcd5ab718f4" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
