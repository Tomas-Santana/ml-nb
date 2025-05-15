import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

df = pd.read_csv("datasets/la_cime.csv")

# Filter to only ASSAULT_WITH_DEADLY_WEAPON since the dataset is too large
ASSAULT_WITH_DEADLY_WEAPON = 230
df = df[df["Crm Cd"] == ASSAULT_WITH_DEADLY_WEAPON]

# Remove outliers (and missing LAT/LON values)
df = df[(df["LAT"] < df["LAT"].mean() + 5 * df["LAT"].std()) & (df["LAT"] > df["LAT"].mean() - 5 * df["LAT"].std())]
df = df[(df["LON"] < df["LON"].mean() + 5 * df["LON"].std()) & (df["LON"] > df["LON"].mean() - 5 * df["LON"].std())]

# Chose 3 clusters based on the elbow method. See plot_elbow_method.py or elbow_method.png for results.
kmeans = KMeans(n_clusters=3, n_init=30)
kmeans.fit(df[["LAT", "LON"]])
df["cluster"] = kmeans.labels_

# Plot the clusters on a map
fig = px.scatter_map(df, lat="LAT", lon="LON", color="cluster")
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(mapbox_center_lat=df["LAT"].mean(), mapbox_center_lon=df["LON"].mean())
fig.update_layout(title="KMeans Clustering of Assaults with Deadly Weapons in LA")
fig.write_html("./src/7_kmeans/imgs/cluster_crime.html")
fig.show()