import pandas as pd

df = pd.read_csv('players.csv')

# Isulate the prediction class
target = df['SEASON']

# Removed columns that will not be used in training (explenation in README.md!!!)
columns_to_drop = ['Unnamed: 0', 'PLAYER', 'SEASON', 'season year']
X = df.drop(columns=columns_to_drop)

print("Columns in our training features (X):")
print(X.columns.tolist())

print(f"\nShape of X: {X.shape}")