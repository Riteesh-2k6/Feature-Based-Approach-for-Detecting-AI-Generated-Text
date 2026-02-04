# verify_robustness.py
import torch
import pandas as pd
from data_loader import load_raid_dataset
from inference import AITextDetector
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

def verify_on_raid(model_path="models/best_model.pt", num_samples=50):
    print(f"Loading model from {model_path}...")
    detector = AITextDetector(model_path)
    
    print("Loading RAID dataset (streaming mode)...")
    from datasets import load_dataset
    try:
        # Using streaming=True to avoid downloading the entire 15GB+ dataset
        dataset = load_dataset("liamdugan/raid", split="extra", streaming=True)
        print("Streaming 'extra' split...")
    except Exception as e:
        print(f"Error streaming extra split: {e}. Trying 'train' split...")
        dataset = load_dataset("liamdugan/raid", split="train", streaming=True)
        
    print(f"Collecting balanced sample of {num_samples} samples...")
    results = []
    
    # Target 50/50 split
    num_human = 0
    num_ai = 0
    target_per_class = num_samples // 2
    
    # Iterate through the streamed dataset
    pbar = tqdm(total=num_samples)
    for row in dataset:
        is_ai = row["model"] != "human"
        
        # Skip if we have enough of this class
        if is_ai and num_ai >= target_per_class:
            continue
        if not is_ai and num_human >= target_per_class:
            continue
            
        text = row["generation"]
        if not text: continue
        
        pred = detector.predict(text)
        results.append({
            "true_label": 1 if is_ai else 0,
            "pred_label": 1 if pred["prediction"] == "AI-Generated" else 0,
            "confidence": pred["confidence"],
            "model": row.get("model", "unknown")
        })
        
        if is_ai: num_ai += 1
        else: num_human += 1
        
        pbar.update(1)
        if num_ai + num_human >= num_samples:
            break
    pbar.close()
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("ROBUSTNESS REPORT (RAID DATASET)")
    print("="*50)
    print(f"Accuracy: {accuracy_score(results_df['true_label'], results_df['pred_label']):.4f}")
    print("\nClassification Report:")
    print(classification_report(results_df["true_label"], results_df["pred_label"]))
    
    print("\nPerformance by Generator Model:")
    model_perf = results_df.groupby("model").apply(lambda x: accuracy_score(x["true_label"], x["pred_label"]))
    print(model_perf)
    
    return results_df

if __name__ == "__main__":
    verify_on_raid()
