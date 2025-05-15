import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("src/6_random_forest_and_bagging_classifiers/random_forest_data/rf_feature_importance.csv")

"""
Visualize the feature importances of the Random Forest classifier trained on the Fashion MNIST dataset.
"""
def plot_importances(df: pd.DataFrame):
    # Get the actual pixel index from the string 'pixel_xxx'
    df['pixel'] = df['pixel'].str.replace('pixel', '', regex=True)
    df = df[df['pixel'] != 'label']
    df['pixel'] = df['pixel'].astype(int)

    # Convert pixel index to x and y coordinates in the 28x28 image
    df['x_pos'] = df['pixel'] % 28
    df['y_pos'] = (df['pixel'] // 28) - 1

    df['importance'] = df['importance'] / df['importance'].max()

    # Create a 28x28 array to hold the importances
    importances = np.zeros((28, 28))

    # Fill the importances array with the values from the DataFrame
    for i in range(len(df)):
        x = int(df.iloc[i]['x_pos']) 
        y = int(df.iloc[i]['y_pos']) 
        importances[y][x] = df.iloc[i]['importance']

    # Plot the importances as a heatmap
    plt.title("Random Forest Feature Importances")
    plt.imshow(importances, cmap='hot', interpolation='nearest')
    plt.savefig("src/6_random_forest_and_bagging_classifiers/imgs/rf_importances.png")
    plt.show()

if __name__ == "__main__":
    plot_importances(df)