# Import the pandas library
import pandas as pd

# Define the path to your downloaded dataset file
# Replace 'path/to/your/spam.csv' with the actual file path on your computer
file_path = 'e:/Cybernaut/Month 2/Email spam(mini project)/spam.csv'

# Use pandas to read the CSV file
# We specify encoding='latin-1' because this particular dataset uses it.
# The file doesn't have a header row, so we'll name the columns 'label' and 'message'.
try:
    df = pd.read_csv(file_path, encoding='latin-1', header=None, names=['label', 'message'])
    
    # Optional: Print the first 5 rows to verify it loaded correctly
    print("Data loaded successfully!")
    print(df.head())

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    print("Please make sure the file path is correct.")