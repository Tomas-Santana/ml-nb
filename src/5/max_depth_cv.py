import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("./datasets/obesity_clean.csv")

y = df.iloc[:, -1].to_numpy()
X = df.iloc[:, :-1].to_numpy()
X = (X - X.mean(axis=0)) / X.std(axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

cv_clf = DecisionTreeClassifier()
param_grid = {"max_depth": range(3, 15)}
grid_search = GridSearchCV(cv_clf, param_grid)
grid_search.fit(X_train, y_train)
clf = grid_search.best_estimator_

score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[["param_max_depth", "mean_test_score"]])


disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
disp.ax_.set_title("Confusion matrix")
plt.show()

# plot max_depth == 3 tree
max_depth_3_clf = DecisionTreeClassifier(max_depth=3)
max_depth_3_clf.fit(X_train, y_train)
plt.figure(figsize=(20, 20))
plot_tree(max_depth_3_clf, filled=True, fontsize=10, feature_names=df.columns[:-1], rounded=True)
plt.title("Decision Tree with max_depth=3")
plt.savefig("./src/5/images/max_depth_3_tree.png", dpi=100)

