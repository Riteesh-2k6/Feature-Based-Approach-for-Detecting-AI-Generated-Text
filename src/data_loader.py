import pandas as pd
from datasets import load_dataset

PATH = "C://REU_Final//data//AI_Human.csv"


def load_kaggle_dataset(path=PATH):
    """Load primary training dataset"""
    df = pd.read_csv(path)
    # Expected columns: 'text', 'label'
    # label: 0 = human, 1 = AI
    print(f"Total samples: {len(df)}")
    print(f"Human samples: {(df['label'] == 0).sum()}")
    print(f"AI samples: {(df['label'] == 1).sum()}")
    return df


def load_raid_dataset(split="train"):
    """Load RAID benchmark dataset using HuggingFace Datasets."""
    print(f"Loading RAID dataset split: {split}")
    raid = load_dataset("liamdugan/raid", encoding="latin-1")
    raid_split = raid[split]
    # Convert to pandas DataFrame for compatibility
    df_raid = pd.DataFrame(raid_split)
    print(f"RAID {split} samples: {len(df_raid)}")
    # Common columns: 'text', 'label', 'attack', 'model', 'domain'
    print("Example entry:\n", df_raid.iloc[0])
    return df_raid


# Usage Example
if __name__ == "__main__":
    # Load Kaggle dataset (local)
    df_kaggle = load_kaggle_dataset(PATH)
    # Load RAID benchmark (from HuggingFace, may take time to download)
    df_raid = load_raid_dataset("train")  # You can also use "test" or "extra"
