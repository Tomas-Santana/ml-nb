import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/magic+gamma+telescope/magic_data_clean.csv")
y = df.iloc[:, -1].to_numpy()
X = df.iloc[:, :-1].to_numpy()
X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rf_clf = RandomForestClassifier()
param_grid = {"n_estimators": [10, 50, 100], "max_depth": range(6, 15), "max_features": [0.5, 0.75, 1]}
grid_search = GridSearchCV(rf_clf, param_grid, verbose=5,)
grid_search.fit(X_train, y_train)
rf_clf = grid_search.best_estimator_
score_df = pd.DataFrame(grid_search.cv_results_)
print("Random Forest Results:")
print(score_df[["param_n_estimators", "param_max_depth", "mean_test_score", "rank_test_score"]])
disp = ConfusionMatrixDisplay.from_estimator(rf_clf, X_test, y_test)
disp.ax_.set_title("Random Forest Confusion matrix")
plt.savefig("src/5/images/rf_cm.png")
