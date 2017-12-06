import numpy as np
from sklearn import datasets, linear_model

from returns_data import read_goog_sp500_data

xData, yData = read_goog_sp500_data()
print(xData, yData)

# set up a linear model to represent this
googModel = linear_model.LinearRegression()

googModel.fit(xData.reshape(-1, 1), yData.reshape(-1, 1))

# find the coeff and intercept of this linear model
print(googModel.coef_)
print(googModel.intercept_)
