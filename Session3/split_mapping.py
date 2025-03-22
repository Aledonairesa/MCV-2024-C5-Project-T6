import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

def split_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, index_col=0)
    
    # Shuffle and split the DataFrame using sklearn
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=False)
    
    # Reset index and drop the old index column
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Generate output file names
    base_name, ext = os.path.splitext(file_path)
    train_file = f"{base_name}_train.csv"
    valid_file = f"{base_name}_validation.csv"
    test_file = f"{base_name}_test.csv"
    
    # Save the splits
    train_df.to_csv(train_file, index=False)
    valid_df.to_csv(valid_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Files saved: {train_file}, {valid_file}, {test_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_mapping.py <file_path>")
        sys.exit(1)
    
    split_csv(sys.argv[1])
