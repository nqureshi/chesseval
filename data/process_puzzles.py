import pandas as pd

# Read CSV 
df = pd.read_csv('lichess_db_puzzle.csv', names=['PuzzleId', 'FEN', 'Moves', 'Rating', 'RatingDeviation', 'Popularity', 'NbPlays', 'Themes', 'GameUrl', 'OpeningTags'])

# Filter for mate in 1
mate1_df = df[df['Themes'].str.contains('mateIn1', na=False)]

# Save filtered CSV
mate1_df.to_csv('lichess_mate1_puzzles.csv', index=False)