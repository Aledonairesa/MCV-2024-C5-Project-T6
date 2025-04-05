import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Clean the FIRD mapping CSV."
    )
    parser.add_argument("--base-path", required=True, help="Base path for FIRD data.")
    parser.add_argument("--input-csv", required=True, help="Path to the raw mapping CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to the cleaned mapping CSV.")

    args = parser.parse_args()

    # Read the raw CSV
    mapping_df = pd.read_csv(args.input_csv)
    
    # Print initial shape
    print(f"Initial DataFrame shape: {mapping_df.shape}")

    # Remove rows where Title is NaN
    mapping_df = mapping_df[~mapping_df["Title"].isna()]

    # Remove rows where Image_Name is "#NAME?"
    mapping_df = mapping_df[mapping_df["Image_Name"] != "#NAME?"]

    # Print final shape
    print(f"Final DataFrame shape: {mapping_df.shape}")

    # Write out the cleaned CSV
    mapping_df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
