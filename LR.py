import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

dir = '/Users/Loielaine/Desktop/umich-2019/EECS545/project/code/545project/'
train = np.loadtxt(dir+'train_sample_no_weather.csv',delimiter=',')
test = np.loadtxt(dir+'test_sample_no_weather.csv',delimiter=',')

X_train = train[:,:-3]
y_train = train[:,-3:]
X_test = test[:,:-3]
y_test = test[:,-3:]

# scaling
def MinMaxScaling(X):
    scaler =preprocessing.MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X

X_train = MinMaxScaling(X_train)
X_test = MinMaxScaling(X_test)

# p20
regr0 = linear_model.LinearRegression()
regr0.fit(X_train, y_train0)
ypred0 = regr0.predict(X_test)
mean_squared_error(y_test[:,0],ypred0)

# p50
regr1 = linear_model.LinearRegression()
regr1.fit(X_train, y_train[:,1])
ypred1 = regr1.predict(X_test)
mean_squared_error(y_test[:,1] ,ypred1)

# p80
regr2 = linear_model.LinearRegression()
regr2.fit(X_train, y_train2)
ypred2 = regr2.predict(X_test)
mean_squared_error(y_test[:,2] ,ypred2)
