# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Add a column to x for the intercept,initialize the theta
2.Perform graadient descent
3.Read the csv file
4.Assuming the last column is ur target variable 'y' and the preceeding column
5.Learn model parameters
6.Predict target value for a new data point

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sharon Harshini L M
RegisterNumber: 212223040193

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

*/
```

## Output:
X & Y VALUES
![3-1](https://github.com/sharon120/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149555539/8983f422-ab6e-4a0d-85cc-c73d4dd00e0c)

X-SCALED & Y-SCALED
![3-2](https://github.com/sharon120/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149555539/779f52fc-b65b-4d55-be09-c5ba9fef86e5)
![3-3](https://github.com/sharon120/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149555539/91ec616d-5a72-4bd5-a44a-adf65c83ddb5)

PREDICTED VALUES
![3-4](https://github.com/sharon120/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/149555539/cf6d95d2-020b-430b-8c65-1d6b2a5571ee)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
