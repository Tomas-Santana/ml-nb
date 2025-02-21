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
'BFH',
'BFD',
'BFA',
'PSH',
'PSD',
'PSA',
]]

cleaned.dropna(inplace=True)
# data normalization
cleaned.iloc[:, 2:] -= np.average(cleaned.iloc[:, 2:], axis=0)
cleaned.iloc[:, 2:] /= np.std(cleaned.iloc[:, 2:], axis=0)

cleaned.to_csv('datasets/premier/20-25_cleaned.csv', index=False)