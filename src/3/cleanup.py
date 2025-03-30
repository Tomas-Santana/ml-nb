import pandas as pd

df = pd.read_csv('./datasets/diabetes_balanced.csv')

print(df.info())
for column in df.columns:
  print(f"{column}: {df[column].unique()}")