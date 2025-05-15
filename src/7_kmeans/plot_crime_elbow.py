from re import A
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/la_cime.csv")

# Filter to only ASSAULT_WITH_DEADLY_WEAPON since the dataset is too large
ASSAULT_WITH_DEADLY_WEAPON = 230
df = df[df["Crm Cd"] == ASSAULT_WITH_DEADLY_WEAPON]

# Remove outliers (and missing LAT/LON values)
df = df[(df["LAT"] < df["LAT"].mean() + 5 * df["LAT"].std()) & (df["LAT"] > df["LAT"].mean() - 5 * df["LAT"].std())]
df = df[(df["LON"] < df["LON"].mean() + 5 * df["LON"].std()) & (df["LON"] > df["LON"].mean() - 5 * df["LON"].std())]

def plot_elbow_method(df, max_k=10):
    inertia = []
    # Calculate inertia for different values of k to find the optimal number of clusters
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=30).fit(df[["LAT", "LON"]])
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")


    plt.xticks(range(1, max_k + 1))
    plt.grid()

    plt.savefig("src/7_kmeans/imgs/elbow_method.png")
    plt.show()

plot_elbow_method(df, max_k=10)