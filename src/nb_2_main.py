import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge as SkRidge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from RegressionLib.Ridge import Ridge as MyRidge
from RegressionLib.LinearRegression import LinearRegression

df = pd.read_csv('datasets/premier/20-25_cleaned.csv')

X = df.iloc[:, 1:].to_numpy()

y = df['FTHG'].to_numpy()

test_size = 0.98

print(f"Absolute largest feature value: {np.max(np.abs(X))}")
print(f"Absolute largest label value: {np.max(np.abs(y))}")


for i in range(20):
  print(f"=======Test {i+1}=======") 
   
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
  reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
  max_reg_coef = np.argmax(np.abs(reg.b_))
  print(f"Max LR coefficient: {reg.b_[max_reg_coef]}")
  rmse = root_mean_squared_error(y_test, reg.predict(X_test))
  print(f"RMSE: {rmse}")
  
  reg = MyRidge(alpha=1.0, fit_intercept=True).fit(X_train, y_train)
  max_reg_coef = np.argmax(np.abs(reg.b_))
  print(f"Max Ridge coefficient: {reg.b_[max_reg_coef]}")
  
  rmse = root_mean_squared_error(y_test, reg.predict(X_test))
  print(f"RMSE: {rmse}")
  
  reg = SkRidge(alpha=1.0, fit_intercept=True).fit(X_train, y_train)
  max_reg_coef = np.argmax(np.abs(reg.coef_))
  
  print(f"Max Sklearn Ridge coefficient: {reg.coef_[max_reg_coef]}")
  rmse = root_mean_squared_error(y_test, reg.predict(X_test))
  print(f"RMSE: {rmse}")
  