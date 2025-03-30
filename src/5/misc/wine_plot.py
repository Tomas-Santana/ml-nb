import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("./datasets/winequality.csv")

df.quality.plot.hist(bins=10)
plt.title("Wine Quality Distribution")
plt.xlabel("Quality")
plt.savefig("wine_quality_distribution.png")
plt.show()

