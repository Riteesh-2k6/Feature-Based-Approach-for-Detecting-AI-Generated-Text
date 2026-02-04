# debug_extractor.py
import pandas as pd
from feature_extractor import LinguisticFeatureExtractor
import sys

def debug_specific_text(row_index):
    print(f"Attempting to debug row {row_index}...")
    try:
        # Read the specific row from the CSV
        df = pd.read_csv("data/train_hardened.csv", skiprows=range(1, row_index), nrows=1)
        if df.empty:
            print("Could not read row. It might be the last line or file is shorter.")
            return
            
        text = df.iloc[0]["text"]
        print("-" * 30)
        print(f"Text Preview (first 200 chars): \n{str(text)[:200]}...")
        print("-" * 30)

        # Run feature extraction
        print("Initializing extractor...")
        extractor = LinguisticFeatureExtractor()
        
        print("Extracting features...")
        features = extractor.extract_all_features(str(text))
        
        print("\n--- SUCCESS ---")
        print(f"Successfully extracted {len(features)} features.")
        print(features)

    except Exception as e:
        print("\n--- FAILURE ---")
        print(f"An error occurred while processing row {row_index}:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        # Optionally, re-raise to get full traceback
        # raise

if __name__ == "__main__":
    # We check row 81364 (the one after the last successful one)
    # The row index in pandas iloc will be 81364, which is the 81365th row of data
    # so we need to skip 81365 rows (1 header + 81364 data rows)
    # Actually, simpler to just read the one row after skipping.
    # skiprows is 1-based for rows after header.
    problematic_index = 81364 
    debug_specific_text(problematic_index)
