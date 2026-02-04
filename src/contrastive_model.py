# contrastive_model.py
import warnings

# Suppress transformers/torch FutureWarning about pytree registration on older
# torch/transformers combinations. This targets the specific message about
# `_register_pytree_node` so other FutureWarnings still appear.
warnings.filterwarnings(
    "ignore",
    message=r".*_register_pytree_node.*",
    category=FutureWarning,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer


class ContrastiveDistilBERT(nn.Module):
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        embedding_dim=768,
        projection_dim=256,
    ):
        super().__init__()

        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )

        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

    def forward(self, input_ids, attention_mask):
        """Forward pass through BERT and projection"""

        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Project for contrastive learning
        projected = self.projection(cls_embedding)

        # L2 normalize for cosine similarity
        projected = F.normalize(projected, p=2, dim=1)

        return cls_embedding, projected

    def encode(self, texts, batch_size=32, device="cpu"):
        """Encode texts to embeddings"""

        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                # Get embeddings
                cls_embedding, _ = self.forward(input_ids, attention_mask)
                all_embeddings.append(cls_embedding.cpu())

        return torch.cat(all_embeddings, dim=0)


class SimCSELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected1, projected2, labels=None):
        """
        Compute contrastive loss

        Args:
            projected1: First view embeddings (batch_size, projection_dim)
            projected2: Second view embeddings (batch_size, projection_dim)
            labels: Optional labels for supervised contrastive learning
        """

        # number of examples in one view (projected1/projected2 have shape
        # (batch_size, projection_dim))
        batch_size = projected1.shape[0]

        # Compute similarity matrix
        # Shape: (2*batch_size, 2*batch_size)
        z = torch.cat([projected1, projected2], dim=0)
        similarity_matrix = torch.matmul(z, z.T) / self.temperature

        # Create mask for positive pairs
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        mask = mask.repeat(2, 2)

        # Remove self-similarities on the big matrix
        mask.fill_diagonal_(False)

        # Positive pairs mask
        pos_mask = torch.zeros_like(mask)
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True

        # Compute loss
        exp_sim = torch.exp(similarity_matrix)

        # Sum over all negatives
        neg_sum = (exp_sim * ~pos_mask).sum(dim=1)

        # Positive similarities
        pos_sim = (exp_sim * pos_mask).sum(dim=1)

        # Contrastive loss
        loss = -torch.log(pos_sim / (pos_sim + neg_sum + 1e-8))

        return loss.mean()


# Usage
model = ContrastiveDistilBERT()
criterion = SimCSELoss(temperature=0.05)
