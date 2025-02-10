import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from lib.LinearRegression import LinearRegression as MyLinearRegression

df = pd.read_csv("datasets/pitch_data_2024_cleaned.csv")


x = np.array(df[["SO", "BB", "WHIP", "W", "L"]]).reshape(-1, 5)
y = np.array(df["ERA"]).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y)

sreg = LinearRegression().fit(x_train, y_train)
myreg = MyLinearRegression.fit(x_train, y_train)

sreg_pred = sreg.predict(x_test)
myreg_pred = myreg.predict(x_test)

print("Sklearn RMSE: ", root_mean_squared_error(y_test, sreg_pred))
print("My RMSE: ", myreg.rmse(y_test, myreg_pred))

print("Sklearn Score: ", sreg.score(x_test, y_test))
print("My Score: ", myreg.score(x_test, y_test))


