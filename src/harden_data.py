# harden_data.py
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import os

def harden_dataset(original_train_path="data/train.csv", output_path="data/train_hardened.csv", raid_samples=5000):
    print(f"Loading original dataset from {original_train_path}...")
    try:
        df_orig = pd.read_csv(original_train_path)
        print(f"Original dataset size: {len(df_orig)}")
    except Exception as e:
        print(f"Error loading original dataset: {e}")
        return

    print(f"Streaming {raid_samples} samples from RAID dataset (balanced)...")
    try:
        dataset = load_dataset("liamdugan/raid", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading RAID: {e}")
        return

    raid_data = []
    num_human = 0
    num_ai = 0
    target_per_class = raid_samples // 2

    pbar = tqdm(total=raid_samples)
    
    for row in dataset:
        if num_human >= target_per_class and num_ai >= target_per_class:
            break

        text = row["generation"]
        if not text: continue
        
        is_ai = row["model"] != "human"
        
        if is_ai:
            if num_ai < target_per_class:
                raid_data.append({"text": text, "label": 1})
                num_ai += 1
                pbar.update(1)
        else:
            if num_human < target_per_class:
                raid_data.append({"text": text, "label": 0})
                num_human += 1
                pbar.update(1)
                
    pbar.close()
    
    df_raid = pd.DataFrame(raid_data)
    print(f"Collected {len(df_raid)} RAID samples.")
    
    print("Merging datasets...")
    # Concatenate without shuffle to preserve cache alignment for the first N rows
    df_hardened = pd.concat([df_orig, df_raid], ignore_index=True)
    # df_hardened = df_hardened.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"New dataset size: {len(df_hardened)}")
    
    print(f"Saving to {output_path}...")
    df_hardened.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    harden_dataset()
