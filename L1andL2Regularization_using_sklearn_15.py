import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# dataset :
dataset = pd.read_csv('melb_data.csv')
print(dataset.head())
print(dataset.nunique())
print(dataset.shape)

# useful columns in the data which on we can use for our task:
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 'Distance', 'CouncilArea', 'Bedroom2', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]
print(dataset.head())
print(dataset.shape)

# checking the null values:
print(dataset.isna().sum())

# making zero to the some columns:
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Car']

# filling na values:
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)
print(dataset.isna().sum())

# Taking out the mean of the building area and apply to the na values:
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())
print(dataset.isna().sum())

# Droping the na values:
dataset.dropna(inplace=True)
print(dataset.isna().sum())

# Creating the dummy variables:
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset.head())

# Dependent variable:
X = dataset.drop('Price', axis=1)
print(X)
# Independent variable:
y = dataset['Price']

# Training and testing the dataset:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

# seleting the best model:
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# In this the accuracy of the test model is -1568 like this
print(reg.score(X_test, y_test))

# the accuracy of this train model is 0.705861 like this
print(reg.score(X_train, y_train))

# Lasso Regreesion (L1):
# there is lot of different in test and train data accuracy:
# for that we use the L1 regularization or Lasso Regression:
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(X_train, y_train)

# this make some better accuracy in the test model:
print("Lasso regression for test model: ",lasso_reg.score(X_test, y_test))

# this make some better accuracy in the train model:
print("Lasso regression for train model: ",lasso_reg.score(X_train, y_train))

# Ridge Regression (L2):
# for that we use the L2 regularization or Ridge Regression:
from sklearn import linear_model
ridge_reg = linear_model.Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(X_train, y_train)

# this make some better accuracy in the test model:
print("Ridge regression for test model: ",ridge_reg.score(X_test, y_test))

# this make some better accuracy in the train model:
print("Ridge regression for train model: ",ridge_reg.score(X_train, y_train))