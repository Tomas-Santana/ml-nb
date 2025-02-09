import pandas as pd

df = pd.read_csv('datasets/pitch_data_2024.csv')

# Drop the columns that are not needed

reduced_df = df[["Player", "Team","SO", "BB", "WHIP", "W", "L", "IP", "ERA"]]

# Drop the rows with missing values

reduced_df = reduced_df.dropna()

# Some players changed teams during the season, so we have rows that show cummulative stats for the player. Team name in these columns is r'\d+TM' where \d+ is the number of teams the player played for. We will drop these rows.

reduced_df = reduced_df = reduced_df[~reduced_df.Team.str.contains(r'\d+TM')]

reduced_df.to_csv('datasets/pitch_data_2024_cleaned.csv', index=False)

