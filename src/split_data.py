# split_data.py
from sklearn.model_selection import train_test_split
import data_loader

PATH = "data/AI_Human.csv"


def create_splits(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Create stratified train/val/test splits"""

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, train_size=train_size, stratify=df["label"], random_state=random_state
    )

    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")

    return train_df, val_df, test_df


if __name__ == "__main__":
    from preprocessing import TextPreprocessor
    import data_loader

    df = data_loader.load_kaggle_dataset(PATH)
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataset(df)

    train_df, val_df, test_df = create_splits(df_clean)

    # Save splits
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
