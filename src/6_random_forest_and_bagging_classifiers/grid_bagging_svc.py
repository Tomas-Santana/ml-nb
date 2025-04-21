from re import X
import pandas as pd
# bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Data from: https://www.kaggle.com/datasets/zalando-research/fashionmnist
df_train = pd.read_csv("datasets/fashion/fashion-mnist_train.csv").sample(1000)
df_test = pd.read_csv("datasets/fashion/fashion-mnist_test.csv")

"""
This is a dataset of 28x28 grayscale images of clothing items, with labels from 0 to 9
0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
"""

X_train = df_train.iloc[:, 1:].to_numpy()
y_train = df_train.iloc[:, 0].to_numpy()

X_test = df_test.iloc[:, 1:].to_numpy()
y_test = df_test.iloc[:, 0].to_numpy()

# We use a linear SVC as the base estimator for the bagging classifier
# and we will use GridSearchCV to find the best hyperparameters for the classifier
bagging_clf = BaggingClassifier(LinearSVC(), n_estimators=50, oob_score=True)
param_grid = {
    "estimator__C": np.logspace(-2, 2, 5), 
    "max_features": [0.5, 0.8, 1.0]
}

# Find the best hyperparameters using GridSearchCV
# We will use a small number of estimators to speed up the process
grid_search = GridSearchCV(bagging_clf, param_grid, verbose=5)
grid_search.fit(X_train, y_train)
bagging_clf = grid_search.best_estimator_

# Save the best hyperparameters and the results to a CSV file
score_df = pd.DataFrame(grid_search.cv_results_)
score_df[["param_estimator__C", "param_max_features", "mean_test_score", "rank_test_score"]].to_csv("src/6/svc_data/bagging_svc_results.csv")

# We will use the best estimator to plot a confusion matrix and save it
disp = ConfusionMatrixDisplay.from_estimator(bagging_clf, X_test, y_test)
disp.ax_.set_title("Bagging SVC Confusion matrix")
plt.savefig("src/6/imgs/bagging_svc_confusion_matrix.png")

print("Train Score: ", bagging_clf.score(X_train, y_train))
print("Test Score: ", bagging_clf.score(X_test, y_test))
print("OOB Score: ", bagging_clf.oob_score_)
