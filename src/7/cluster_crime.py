import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

df = pd.read_csv("datasets/la_cime.csv")

print(df["LAT"].describe())
print(df["LON"].describe())

# Filter to only ASSAULT_WITH_DEADLY_WEAPON since the dataset is too large
VEHICULE_THEFT = 510
ASSAULT_WITH_DEADLY_WEAPON = 230
df = df[df["Crm Cd"] == ASSAULT_WITH_DEADLY_WEAPON]

df = df[(df["LAT"] < df["LAT"].mean() + 5 * df["LAT"].std()) & (df["LAT"] > df["LAT"].mean() - 5 * df["LAT"].std())]
df = df[(df["LON"] < df["LON"].mean() + 5 * df["LON"].std()) & (df["LON"] > df["LON"].mean() - 5 * df["LON"].std())]

kmeans = KMeans(n_clusters=6)
kmeans.fit(df[["LAT", "LON"]])
df["cluster"] = kmeans.labels_

fig = px.scatter_map(df, lat="LAT", lon="LON", color="cluster")
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(mapbox_zoom=10, mapbox_center_lat=df["LAT"].mean(), mapbox_center_lon=df["LON"].mean())
fig.show()