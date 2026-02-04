# analyze_features.py
import pandas as pd
import numpy as np
import os
from feature_extractor import LinguisticFeatureExtractor

def get_feature_names():
    print("Initializing extractor to get feature names...")
    extractor = LinguisticFeatureExtractor()
    # Dummy text
    dummy = "This is a dummy text to extract feature names. It needs to be long enough to generate all features."
    features_dict = extractor.extract_all_features(dummy, return_dict=True)
    return list(features_dict.keys())

def analyze_feature_importance():
    print("Loading data...")
    try:
        val_df = pd.read_csv("data/val.csv")
        val_features = np.load("data/val_features.npy")
    except FileNotFoundError:
        print("Error: data/val.csv or data/val_features.npy not found.")
        return

    print(f"Loaded {len(val_df)} validation samples.")
    print(f"Loaded {val_features.shape} feature matrix.")
    
    # Ensure alignment
    if len(val_df) != len(val_features):
        print(f"Warning: Length mismatch. CSV: {len(val_df)}, Features: {len(val_features)}")
        min_len = min(len(val_df), len(val_features))
        val_df = val_df.iloc[:min_len]
        val_features = val_features[:min_len]

    # Get feature names
    feature_names = get_feature_names()
    
    if len(feature_names) != val_features.shape[1]:
        print(f"Error: Feature dimension mismatch. Names: {len(feature_names)}, Array: {val_features.shape[1]}")
        # Try to fix if it's just a few missing
        if val_features.shape[1] > len(feature_names):
             print("Array has more features. Truncating array for analysis...")
             val_features = val_features[:, :len(feature_names)]
        else:
             print("Names list is longer. Truncating names...")
             feature_names = feature_names[:val_features.shape[1]]

    # Create DataFrame
    print("Calculating correlations...")
    feat_df = pd.DataFrame(val_features, columns=feature_names)
    feat_df["label"] = val_df["label"].values # 0=Human, 1=AI
    
    # Calculate correlation with label
    correlations = feat_df.corr()["label"].drop("label")
    
    # Create summary
    summary_df = pd.DataFrame({
        "Feature": correlations.index,
        "Correlation": correlations.values,
        "AbsCorrelation": correlations.abs()
    })
    
    summary_df = summary_df.sort_values("AbsCorrelation", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS (Correlation with 'AI' Label)")
    print("="*70)
    print(f"{ 'Rank':<5} | { 'Feature Name':<35} | { 'Correlation':<10}")
    print("-" * 70)
    
    for i in range(min(25, len(summary_df))):
        row = summary_df.iloc[i]
        print(f"{i+1:<5} | {row['Feature']:<35} | {row['Correlation']:<10.4f}")
        
    print("-" * 70)
    print("Interpretation:")
    print("Correlation > 0: Feature is HIGHER in AI-generated text.")
    print("Correlation < 0: Feature is HIGHER in Human-written text.")
    
    # Save
    summary_df.to_csv("feature_importance.csv", index=False)
    print("\nReport saved to feature_importance.csv")

if __name__ == "__main__":
    analyze_feature_importance()
