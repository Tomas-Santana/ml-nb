import pandas as pd
import numpy as np

data_path = 'datasets/premier/{season}.csv'
# we will use the same dataframes for all the seasons
dfs = {}

for season in range(20, 26):
    dfs[season] = pd.read_csv(data_path.format(season=season))

# concatenate all the dataframes
df = pd.concat(dfs.values(), ignore_index=True, axis=0)

cleaned = df[[
'FTHG',
'FTAG',
'HTHG',
'HTAG',
'HS',
'AS',
'HST',
'AST',
'HF',
'AF',
'HC',
'AC',
'HY',
'AY',
'HR',
'AR', 
'MaxH', 
'MaxD', 
'MaxA', 
'AvgH', 
'AvgD', 
'AvgA',
'B365H',
'B365D',
'B365A',
'PSH',
'PSD',
'PSA',
'WHH',
'WHD',
'WHA',
'Max>2.5',
'Max<2.5',
'Avg>2.5',
'Avg<2.5',
'AHh',
'B365AHH',
'B365AHA',
'PAHH',
'PAHA',
'MaxAHH',
'MaxAHA',
]]

# cast all int64 to float64
cleaned = cleaned.astype('float64')

cleaned.dropna(inplace=True)
# data normalization
cleaned.iloc[:, 1:] -= np.average(cleaned.iloc[:, 1:], axis=0)
cleaned.iloc[:, 1:] /= np.std(cleaned.iloc[:, 1:], axis=0)

cleaned.to_csv('datasets/premier/20-25_cleaned.csv', index=False)

# number of rows
print("Data points:", cleaned.shape[0])