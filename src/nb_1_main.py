"""
Data from https://baseball-reference.com
https://www.baseball-reference.com/leagues/majors/2024-standard-pitching.shtml
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from RegressionLib.LinearRegression import LinearRegression as MyLinearRegression
from RegressionLib.Ridge import Ridge as MyRidge

df = pd.read_csv("datasets/pitch_data_2024_cleaned.csv")

# Reshape the data to 2D arrays
x = np.array(df[["SO/BB", "IP", "WHIP", "W", "L"]]).reshape(-1, 5)

y = np.array(df["ERA"]).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y)

myreg = MyLinearRegression().fit(x_train, y_train)
reg = LinearRegression().fit(x_train, y_train)
ridge = Ridge().fit(x_train, y_train)
myridge = MyRidge().fit(x_train, y_train)

myreg_pred = myreg.predict(x_test)
reg_pred = reg.predict(x_test)
ridge_pred = ridge.predict(x_test)
myridge_pred = myridge.predict(x_test)


print("Sklearn Linear Regression RMSE: ", root_mean_squared_error(y_test, reg_pred))
print("My Linear Regression RMSE: ", myreg.rmse(y_test, myreg_pred))
print("Sklearn Ridge RMSE: ", root_mean_squared_error(y_test, ridge_pred))
print("My Ridge RMSE: ", myridge.rmse(y_test, myridge_pred))


print("Sklearn Linear Regression R^2: ", reg.score(x_test, y_test))
print("My Linear Regression R^2: ", myreg.score(x_test, y_test))
print("Sklearn Ridge R^2: ", ridge.score(x_test, y_test))
print("My Ridge R^2: ", myridge.score(x_test, y_test))



