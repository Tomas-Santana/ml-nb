import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt

df = pd.read_csv("./datasets/premier/20-25_cleaned.csv")

# Selecting features and labels
X = df.iloc[:, 1:].to_numpy()
y = df['FTHG'].to_numpy()

# Normalizing data
X = (X-np.average(X, axis=0))/np.std(X, axis=0)
y = (y-np.average(y)) / np.std(y)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lasso = Lasso(max_iter=10000)
params = {"alpha": np.linspace(0.005, 0.008, num=20)}
gs = GridSearchCV(lasso, param_grid=params, cv=5, scoring="r2",)
gs.fit(X_train, y_train)
score_df = pd.DataFrame(gs.cv_results_)
print(f"=============== Grid Search Results: ===============")
print(score_df[['param_alpha', 'mean_test_score', 'rank_test_score']])


best_lasso = Lasso(alpha=gs.best_params_["alpha"])
best_lasso.fit(X_train, y_train)

print(f"R^2 of Test Data (alpha={gs.best_params_['alpha']:.4f}): {best_lasso.score(X_test, y_test):.3f}")

print(best_lasso.coef_)

score_df.plot.line(x='param_alpha', y='mean_test_score')
plt.scatter(gs.best_params_['alpha'], gs.best_score_, color='red', label=f'Best Score: {gs.best_score_:.3f} at alpha={gs.best_params_["alpha"]:.4f}')
plt.legend()
plt.show()
