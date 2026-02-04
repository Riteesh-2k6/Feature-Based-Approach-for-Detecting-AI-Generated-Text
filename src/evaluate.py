# train.py
import torch
from feature_extractor import LinguisticFeatureExtractor
from hybrid_model import HybridAIDetector
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
    def __init__(self, df, feature_extractor, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-extract linguistic features to save time
        print("Extracting linguistic features...")
        self.linguistic_features = []
        for idx in tqdm(range(len(df))):
            text = df.iloc[idx]["text"]
            features = feature_extractor.extract_all_features(text)
            self.linguistic_features.append(features)
        self.linguistic_features = np.array(self.linguistic_features)

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


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
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

    # Set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Load data
    print("Loading data...")
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    # Initialize feature extractor
    print("Initializing feature extractor...")
    feature_extractor = LinguisticFeatureExtractor()

    # Initialize model
    print("Initializing model...")
    model = HybridAIDetector()
    model = model.to(config["device"])

    # Create datasets
    print("Creating datasets...")
    train_dataset = AIDetectionDataset(
        train_df, feature_extractor, model.embedding_model.tokenizer
    )
    val_dataset = AIDetectionDataset(
        val_df, feature_extractor, model.embedding_model.tokenizer
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=4)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    best_val_f1 = 0

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'=' * 50}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, config["device"], epoch
        )
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

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
