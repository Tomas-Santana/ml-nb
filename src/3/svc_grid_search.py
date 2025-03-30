from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# dataset from https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
df = pd.read_csv("./datasets/diabetes_balanced.csv")
print(df.iloc[:, 0].value_counts().apply(lambda x: x / len(df)))

X = df.iloc[:, 1:].to_numpy()

y = df.iloc[:, 0].to_numpy()

X = (X - X.mean()) / X.std()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

svc = LinearSVC()
params = {"C": np.logspace(-2, 2, 5)}
gs = GridSearchCV(svc, param_grid=params, cv=5, scoring="accuracy", verbose=50)
gs.fit(X_train, y_train)
score_df = pd.DataFrame(gs.cv_results_)
print(f"=============== Grid Search Results: ===============")
print(score_df[['param_C', 'mean_test_score', 'rank_test_score']])
print(f"Best C: {gs.best_params_['C']}")

clf = gs.best_estimator_
display = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()


score_df.plot.line(x='param_C', y='mean_test_score')
plt.scatter(gs.best_params_['C'], gs.best_score_, color='red', label=f'Best Score: {gs.best_score_:.3f} at C={gs.best_params_["C"]:.4f}')
plt.legend()
plt.show()







