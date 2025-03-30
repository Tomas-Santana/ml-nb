import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge as SkRidge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from RegressionLib.Ridge import Ridge as MyRidge
import matplotlib.pyplot as plt
from RegressionLib.LinearRegression import LinearRegression

df = pd.read_csv('datasets/premier/20-25_cleaned.csv')

X = df.iloc[:, 1:].to_numpy()

y = df['FTHG'].to_numpy()

test_size = 0.98

print(f"Absolute largest feature value: {np.max(np.abs(X))}")
print(f"Absolute largest label value: {np.max(np.abs(y))}")


max_coef_df = pd.DataFrame(columns=['LR', 'Ridge', 'Sklearn Ridge'])
iters = 500

for i in range(iters):
  print(f"=======Test {i+1}=======") 
   
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
  lr_reg = LinearRegression(fit_intercept=False).fit(X_train, y_train)
  lr_max_reg_coef = np.argmax(np.abs(lr_reg.b_))
  print(f"Max LR coefficient: {lr_reg.b_[lr_max_reg_coef]}")
  r2 = lr_reg.score(X_test, y_test)
  print(f"R2: {r2}")

  
  mr_reg = MyRidge(alpha=1.0, fit_intercept=True).fit(X_train, y_train)
  mr_max_reg_coef = np.argmax(np.abs(mr_reg.b_))
  print(f"Max Ridge coefficient: {mr_reg.b_[mr_max_reg_coef]}")

  
  r2 = mr_reg.score(X_test, y_test)
  print(f"R2: {r2}")
  
  sr_reg = SkRidge(alpha=1.0, fit_intercept=True).fit(X_train, y_train)
  sr_max_reg_coef = np.argmax(np.abs(sr_reg.coef_))
  
  print(f"Max Sklearn Ridge coefficient: {sr_reg.coef_[sr_max_reg_coef]}")
  r2 = sr_reg.score(X_test, y_test)
  print(f"R2: {r2}")

  max_coef_df.loc[-1] = [lr_reg.b_[lr_max_reg_coef], mr_reg.b_[mr_max_reg_coef], sr_reg.coef_[sr_max_reg_coef]]
  max_coef_df.index = max_coef_df.index + 1
  

# histogram of the maximum coefficient values

max_coef_df["LR"].plot.hist(alpha=0.5, bins=20, color='r', label='LR')
plt.legend("LR")
plt.show()
max_coef_df["Ridge"].plot.hist(alpha=0.5, bins=20, color='b', label='Ridge')
plt.legend("Ridge")
plt.show()