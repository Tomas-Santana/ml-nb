import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("datasets/magic+gamma+telescope/magic_data_clean.csv")
y = df.iloc[:, -1].to_numpy()
X = df.iloc[:, :-1].to_numpy()
X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


svc_clf = SVC()
param_grid = {"C": np.logspace(-2, 2, 5), "gamma": np.logspace(-2, 2, 5)}
grid_search = GridSearchCV(svc_clf, param_grid, verbose=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
svc_clf = grid_search.best_estimator_

score_df = pd.DataFrame(grid_search.cv_results_)
print("SVC Results:")
print(score_df[[ "param_C", "param_gamma", "mean_test_score", "rank_test_score"]])
disp = ConfusionMatrixDisplay.from_estimator(svc_clf, X_test, y_test)
disp.ax_.set_title("SVC Confusion matrix")
plt.savefig("src/5/images/svc_cm.png")



