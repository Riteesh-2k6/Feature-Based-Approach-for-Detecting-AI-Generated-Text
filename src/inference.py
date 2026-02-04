# inference.py
import torch
from feature_extractor import LinguisticFeatureExtractor
from hybrid_model import HybridAIDetector


class AITextDetector:
    def __init__(self, model_path, device="cpu"):
        self.device = device

        # Load feature extractor
        self.feature_extractor = LinguisticFeatureExtractor()

        # Load model
        self.model = HybridAIDetector(feature_dim=54)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(device)
        self.model.eval()

    def predict(self, text):
        """
        Predict if text is AI-generated

        Returns:
            dict with 'prediction', 'confidence', 'probabilities'
        """
        # Extract features
        ling_features = self.feature_extractor.extract_all_features(text)
        ling_features = torch.FloatTensor(ling_features).unsqueeze(0)

        # Get prediction
        predictions, confidences = self.model.predict(
            ling_features, [text], device=self.device
        )

        pred_class = predictions.item()
        confidence = confidences.item()

        return {
            "prediction": "AI-Generated" if pred_class == 1 else "Human-Written",
            "confidence": confidence,
            "ai_probability": confidence if pred_class == 1 else 1 - confidence,
        }


# Usage
detector = AITextDetector("models/best_model.pt")
result = detector.predict("Your text here...")
print(result)
