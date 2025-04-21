import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Data from: https://www.kaggle.com/datasets/zalando-research/fashionmnist
# This is a dataset of 28x28 grayscale images of clothing items, with labels from 0 to 9
df_train = pd.read_csv("datasets/fashion/fashion-mnist_train.csv").sample(1000)
df_test = pd.read_csv("datasets/fashion/fashion-mnist_test.csv")

X_train = df_train.iloc[:, 1:].to_numpy()
y_train = df_train.iloc[:, 0].to_numpy()

X_test = df_test.iloc[:, 1:].to_numpy()
y_test = df_test.iloc[:, 0].to_numpy()

# We will use a Random Forest classifier and we will use GridSearchCV to find the best hyperparameters for the classifier
rf_clf = RandomForestClassifier(oob_score=True, max_features=0.5, n_jobs=-1, )
param_grid = {
    "n_estimators": [100, 150], 
    "max_depth": range(5, 10), 
    "max_features": [0.5, 0.8, 1.0]
}

# Find the best hyperparameters using GridSearchCV
grid_search = GridSearchCV(rf_clf, param_grid, verbose=5)
grid_search.fit(X_train, y_train)

rf_clf = grid_search.best_estimator_

# Save the best hyperparameters and the results to a CSV file
score_df = pd.DataFrame(grid_search.cv_results_)
score_df[["param_n_estimators", "param_max_depth", "mean_test_score", "rank_test_score"]].to_csv("src/6/random_forest_data/rf_results.csv")

# Plot the confusion matrix using the best estimator and save it
disp = ConfusionMatrixDisplay.from_estimator(rf_clf, X_test, y_test)
disp.ax_.set_title("Random Forest Confusion matrix")
plt.savefig("src/6/imgs/rf_confusion_matrix.png")

print("Train Score: ", rf_clf.score(X_train, y_train))
print("Test Score: ", rf_clf.score(X_test, y_test))
print("OOB Score: ", rf_clf.oob_score_)

# Get the feature importances and save them to a CSV file (We will use this to visualize the feature importances)
imp_df = pd.DataFrame(rf_clf.feature_importances_, columns=["importance"])
imp_df.index = df_test.columns[1:]
imp_df.index.name = "pixel"

imp_df.to_csv("src/6/random_forest_data/rf_feature_importance.csv")