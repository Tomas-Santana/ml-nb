import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("./datasets/obesity_clean.csv")

y = df.iloc[:, -1].to_numpy()
X = df.iloc[:, :-1].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Decision Tree Classifier. Grid search for max_depth
cv_clf = DecisionTreeClassifier()
param_grid = {"max_depth": range(3, 15)}
grid_search = GridSearchCV(cv_clf, param_grid)
grid_search.fit(X_train, y_train)
clf = grid_search.best_estimator_

score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[["param_max_depth", "mean_test_score", "rank_test_score"]])

fig, ax = plt.subplots(figsize=(10, 10))

disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax)
disp.ax_.set_title("Confusion matrix for Decision Tree Classifier")


for label in disp.ax_.get_xticklabels():
    label.set_fontsize(8)
    label.set_rotation(45)
for label in disp.ax_.get_yticklabels():
    label.set_fontsize(8)

plt.savefig("./src/5_decision_trees_random_forest/imgs/decision_tree_confusion_matrix.png", dpi=100)


# Max depth 3 decision tree
class_names = sorted(df.iloc[:, -1].unique())
max_depth_3_clf = DecisionTreeClassifier(max_depth=3)
max_depth_3_clf.fit(X_train, y_train)
plt.figure(figsize=(20, 20))
plot_tree(max_depth_3_clf, filled=True, fontsize=10, feature_names=df.columns[:-1], class_names=class_names, rounded=True)
plt.title("Decision Tree with max_depth=3")
plt.savefig("./src/5_decision_trees_random_forest/imgs/max_depth_3_tree.png")
