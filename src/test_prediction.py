# test_prediction.py
import torch
from inference import AITextDetector
import sys

def main():
    text = """I just mean- doesn't it bother you? Do you know what the upper years call him?" Pansy asked. Theo gestured for her to continue. "The Unseen Potter. It's because no one knows what he's doing. No one sees him outside meal times, class, and quidditch. Slytherin is the house of cunning and ambition, but you guys all seem content following someone who's seemingly never around."

"Because he's busy being the bloody Heritor of Slytherin," Theo answered flatly. "Or did you forget that?"""
    
    print("Initializing detector...")
    try:
        detector = AITextDetector("models/best_model.pt", device="cuda")
        print("\nAnalyzing text...")
        result = detector.predict(text)
        
        print("-" * 30)
        print(f"Text: {text[:100]}...")
        print("-" * 30)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"AI Probability: {result['ai_probability']:.4f}")
        print("-" * 30)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

