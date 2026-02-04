# test_prediction_4.py
import torch
from inference import AITextDetector

def main():
    text = """Too much formality, no bending of the rules, strict adherence to grammatical rules, and above all a lack of colourful phrasing and original metaphors are a strong indicator that an AI has done it.

Unfortunately the above are also prevalent where somebody none too good at the craft of wordsmithery has done it too. So put in the odd original phrase, highly convoluted sentence, and ridiculous flight of fancy, and no bugger will ever credit that a feckin’ robot wrote it, innit?

I’d stick in lexemes that are grandiosely magniloquent as well, especially if the last time they saw combat was before eighteen ninety-six, such as epicaricacy, obequitate, septentrional, accismus, and it never hurts to create your own portmanteaus such as bintybird (a word I have recently taken to using to describe human females since we are not allowed to call them women any more) which the bots never do."""
    
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
