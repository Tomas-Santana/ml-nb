import pandas as pd

names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("./datasets/magic+gamma+telescope/magic04.data", names=names)

df.to_csv("./datasets/magic_data_clean.csv", index=False)
print(df.head())