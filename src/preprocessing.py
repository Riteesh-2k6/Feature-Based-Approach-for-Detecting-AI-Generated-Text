# preprocessing.py
import re
import pandas as pd
import data_loader
PATH = "C://REU_Final//data//AI_Human.csv"
df = data_loader.load_kaggle_dataset(PATH)
class TextPreprocessor:
    def __init__(self, min_length=50, max_length=5000):
        self.min_length = min_length
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"()\[\]]', "", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def filter_by_length(self, text: str) -> bool:
        """Check if text meets length requirements"""
        length = len(text.split())
        return self.min_length <= length <= self.max_length

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataset"""
        # Clean texts
        df["text"] = df["text"].apply(self.clean_text)

        # Filter by length
        df["valid_length"] = df["text"].apply(self.filter_by_length)
        df = df[df["valid_length"]].drop("valid_length", axis=1)

        # Remove duplicates
        df = df.drop_duplicates(subset=["text"])

        # Reset index
        df = df.reset_index(drop=True)

        print(f"After preprocessing: {len(df)} samples")
        return df


# Usage
preprocessor = TextPreprocessor()
df_clean = preprocessor.preprocess_dataset(df)

# Add this line to save the result
df_clean.to_csv("data/cleaned_data.csv", index=False)
