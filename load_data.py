import pandas as pd

df = pd.read_csv('players.csv')

# Display the first 5 rows to see what the data looks like
print("First 5 rows of the dataset:")
print(df.head())

# Show information about the columns, data types, and missing values
print("\nDataset Information:")
print(df.info())

# Check the balance of our target label (SEASON)
print("\nTarget Label Distribution:")
print(df['SEASON'].value_counts())