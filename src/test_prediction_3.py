# test_prediction_3.py
import torch
from inference import AITextDetector

def main():
    text = """People can only accurately identify AI-generated text about 50% of the time, which is roughly equivalent to random chance.
 This finding comes from a study conducted by Jeff Hancock and his team at Stanford University, which evaluated text from platforms like OKCupid, Airbnb, and Guru.com, showing participants could not reliably distinguish between human and AI writing.
 The research indicates that individuals rely on flawed heuristics—such as associating high grammatical correctness, first-person pronouns, family references, and informal language with human authorship—leading to consistent but incorrect judgments.
 Despite the widespread use of AI in content creation, including in online dating, professional contexts, and hospitality, detection remains unreliable.
 Some AI detection tools claim high accuracy, such as GPTZero reporting a 99% accuracy rate in spotting AI text, but these claims are contested, and many experts emphasize that no tool can definitively determine if AI was used"""
    
    print("Initializing detector...")
    try:
        detector = AITextDetector("models/best_model.pt", device="cpu")
        print("\nAnalyzing text...")
        result = detector.predict(text)
        
        print("-" * 30)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"AI Probability: {result['ai_probability']:.4f}")
        print("-" * 30)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

