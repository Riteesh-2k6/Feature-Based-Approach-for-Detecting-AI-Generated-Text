# hybrid_model.py
import torch
import torch.nn as nn
from contrastive_model import ContrastiveDistilBERT
from attention_fusion import IterativeAFF

class HybridAIDetector(nn.Module):
    def __init__(self, feature_dim=57, embedding_dim=768, fused_dim=256, num_classes=2):
        super().__init__()

        # Embedding model
        self.embedding_model = ContrastiveDistilBERT()

        # Fusion module
        self.fusion = IterativeAFF(feature_dim, embedding_dim, fused_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, linguistic_features, input_ids, attention_mask):
        """
        Args:
            linguistic_features: (batch_size, 57)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Get embeddings from DistilBERT
        embeddings, _ = self.embedding_model(input_ids, attention_mask)

        # Fuse features
        fused = self.fusion(linguistic_features, embeddings)

        # Classify
        logits = self.classifier(fused)

        return logits

    def predict(self, linguistic_features, texts, device="cpu"):
        """Make predictions with confidence scores"""

        self.eval()

        # Tokenize texts
        encoded = self.embedding_model.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        linguistic_features = linguistic_features.to(device)

        with torch.no_grad():
            logits = self.forward(linguistic_features, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            confidences = torch.max(probs, dim=1).values

        return predictions.cpu(), confidences.cpu()


# Usage
model = HybridAIDetector()
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
