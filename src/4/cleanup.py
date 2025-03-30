import pandas as pd


name_file = open('datasets/spambase/spambase.names', 'r')


columns = []
# Get column names from the spambase.names file
for line in name_file:
  if line.startswith('|') or line.isspace() or line.startswith('1'):
    continue
  column_name = line.split(':')[0]
  columns.append(column_name)

columns.append('class')

df = pd.read_csv('datasets/spambase/spambase.data', names=columns)

print(df.info())

df.to_csv('datasets/spambase/spambase_clean.csv', index=False)