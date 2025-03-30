import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():

  df = pd.read_csv("./datasets/spambase/spambase_clean.csv")

  X = df.iloc[:, 1:].to_numpy()

  y = df.loc[:, "class"].to_numpy()

  X = (X - X.mean()) / X.std()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  params = {
    "C": np.logspace(-2, 2, 5),
    "gamma": np.logspace(-2, 2, 5)
  }
  scorings = ['accuracy', "recall", "f1"]
  
  for scoring in scorings:
    svc = SVC(kernel="rbf")
    gs = GridSearchCV(svc, params, cv=2, scoring=scoring)
    gs.fit(X_train, y_train)
 
    score_df = pd.DataFrame(gs.cv_results_)
    print(f"=============== {scoring} Grid Search Results: ===============")
    print(score_df[['param_C', 'param_gamma', 'mean_test_score', 'rank_test_score']])

    best_svc = gs.best_estimator_
    
    ConfusionMatrixDisplay.from_estimator(best_svc, X_test, y_test, normalize="true")
    plot_scores_3d(f"Grid Search {scoring}", score_df["mean_test_score"], score_df["param_C"], score_df["param_gamma"], "C", "gamma", gs.best_score_, gs.best_params_["C"], gs.best_params_["gamma"])
    plt.show()

def plot_scores_3d(title, test_score, x, y, x_label, y_label, test_score_best=None,x_best=None, y_best=None):
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, y, test_score)
  
  if (x_best and y_best and test_score_best):
    ax.scatter(x_best, y_best, test_score_best, color='red')
  ax.set_title(title)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_zlabel('mean_test_score')
  plt.show()
  
if __name__ == "__main__":
  main()






