# train.py
import os
import sys
from feature_extractor import LinguisticFeatureExtractor
from hybrid_model import HybridAIDetector
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


class AIDetectionDataset(Dataset):
    def __init__(self, df, feature_extractor, model_tokenizer, max_length=512, cache_name="train"):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.tokenizer = model_tokenizer # Renamed for clarity
        self.max_length = max_length
        self.cache_path = f"data/{cache_name}_features.npy"

        # Pre-extract linguistic features with checkpointing
        features_list = []
        start_idx = 0
        
        if os.path.exists(self.cache_path):
            print(f"Loading features from cache: {self.cache_path}")
            try:
                cached_features = np.load(self.cache_path)
                if len(cached_features) == len(self.df):
                    self.linguistic_features = cached_features
                    print("Cache complete.")
                    return
                elif len(cached_features) < len(self.df):
                    print(f"Cache partial: found {len(cached_features)}/{len(self.df)}. Resuming...")
                    features_list = list(cached_features)
                    start_idx = len(cached_features)
                else:
                    print(f"Cache mismatch (too large): expected {len(self.df)}, found {len(cached_features)}. Regenerating...")
            except Exception as e:
                print(f"Error loading cache: {e}. Regenerating...")

        print(f"Extracting linguistic features from index {start_idx}...")
        for idx in tqdm(range(start_idx, len(self.df))):
            try:
                text = self.df.iloc[idx]["text"]
                # Handle potential NaN/empty text
                if not isinstance(text, str) or text.strip() == "":
                    features = np.zeros(54)
                else:
                    features = self.feature_extractor.extract_all_features(text)
                
                # Final check on feature dimension
                if features.shape[0] != 54:
                    print(f"Warning: Feature dimension mismatch at index {idx}. Got {features.shape[0]}, expected 54. Using zeros.")
                    features = np.zeros(54)
                    
            except Exception as e:
                print(f"\n--- WARNING: Error processing text at index {idx}. Skipping. ---")
                print(f"Error: {e}")
                features = np.zeros(54) # Append a zero vector
                
            features_list.append(features)

            # Save progress every 1000 features
            if (idx + 1) % 1000 == 0:
                np.save(self.cache_path, np.array(features_list))
        
        self.linguistic_features = np.array(features_list)
        np.save(self.cache_path, self.linguistic_features)
        print(f"Saved features to cache: {self.cache_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ling_features = torch.FloatTensor(self.linguistic_features[idx])

        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "linguistic_features": ling_features,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long),
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, start_step=0, best_val_f1=0):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    # If resuming, skip already processed batches
    if start_step > 0:
        print(f"Skipping first {start_step} batches...")
        # We don't update progress bar here to avoid visual confusion, 
        # instead we just let the loop skip and then the bar will start updating
    
    last_step = start_step 

    for i, batch in enumerate(progress_bar):
        if i < start_step:
            continue
            
        # Move to device
        ling_feat = batch["linguistic_features"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(ling_feat, input_ids, attention_mask)

        # Calculate loss
        loss = nn.CrossEntropyLoss()(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})
        
        # Explicit print for monitoring
        if (i + 1) % 10 == 0:
             print(f"Step {i+1}, Loss: {loss.item():.4f}")

        # --- Intra-epoch Checkpointing ---
        if (i + 1) % 1000 == 0:
            checkpoint_path = f"models/training_checkpoint.pt"
            print(f"\nSaving intra-epoch checkpoint at step {i+1}...")
            torch.save({
                'epoch': epoch,
                'step': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': best_val_f1
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")



    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            ling_feat = batch["linguistic_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(ling_feat, input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    auc = roc_auc_score(all_labels, all_probs)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def main():
    # Configuration
    config = {
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 5,
        "max_length": 512,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "models/",
        "seed": 42,
    }

    # Create directories if they don't exist
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs("data", exist_ok=True)


    # Set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Load data
    print("Loading data...")
    train_df = pd.read_csv("data/train_hardened.csv")
    val_df = pd.read_csv("data/val.csv")

    # Initialize feature extractor
    print("Initializing feature extractor...")
    feature_extractor = LinguisticFeatureExtractor()

    # Initialize model
    print("Initializing model...")
    model = HybridAIDetector(feature_dim=54)
    model = model.to(config["device"])

    # Create datasets
    print("Creating datasets...")
    train_dataset = AIDetectionDataset(
        train_df, feature_extractor, model.embedding_model.tokenizer, cache_name="train_hardened"
    )
    val_dataset = AIDetectionDataset(
        val_df, feature_extractor, model.embedding_model.tokenizer, cache_name="val"
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=0)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # --- Checkpoint Loading ---
    start_epoch = 1
    start_step = 0
    best_val_f1 = 0
    checkpoint_path = f"{config['save_dir']}/training_checkpoint.pt"

    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        best_val_f1 = checkpoint.get('val_f1', 0)
        
        if 'step' in checkpoint:
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step'] + 1
            print(f"Resuming from epoch {start_epoch}, step {start_step} with best F1: {best_val_f1:.4f}")
        else:
            start_epoch = checkpoint['epoch'] + 1
            start_step = 0
            print(f"Resuming from epoch {start_epoch} with best F1: {best_val_f1:.4f}")


    # Training loop
    for epoch in range(start_epoch, config["num_epochs"] + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'=' * 50}")

        # Train
        # If resuming mid-epoch, we need a way to skip steps. 
        # A simple way is to pass start_step to train_epoch.
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, config["device"], epoch, start_step if epoch == start_epoch else 0, best_val_f1
        )
        start_step = 0 # Reset after first resumed epoch
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, config["device"])
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f}")

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                    "config": config,
                },
                f"{config['save_dir']}/best_model.pt",
            )
            print(f"* Saved best model (F1: {best_val_f1:.4f})")

        # --- Save Training Checkpoint ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': best_val_f1
        }, checkpoint_path)
        print(f"* Saved training checkpoint to {checkpoint_path}")


    print("\nTraining complete!")


if __name__ == "__main__":
    main()
