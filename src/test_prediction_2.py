# test_prediction_2.py
import torch
from inference import AITextDetector

def main():
    text = """Pansy gasped softly, eyes widening like she’d just realized something very deep. “Ten steps ahead?” she whispered.

    Theo’s gaze darkened, shadows dancing across his sharp features from the flickering green fire. “Always,” he said ominously. “Potter doesn’t play chess. He is the board.”

    A sudden hush fell over the common room, as if the very walls were listening.

    “He walks the corridors unseen,” Blaise continued smoothly, swirling his drink even though it was empty, “speaks to ghosts that don’t answer to professors, and studies magic older than Hogwarts itself.”

    Daphne shivered. “I heard he doesn’t even sleep. That the heir magic sustains him.”

    Pansy swallowed. “That’s— that’s insane.”

    Just then, the torches flared violently.

    A cold wind swept through the dungeon, despite there being no windows.

    Theo smirked. “Funny thing about talking about Potter,” he drawled. “He tends to notice.”

    From the shadows near the entrance, a tall figure emerged, cloak billowing even though there was no breeze now. Green eyes gleamed like emerald fire beneath the hood.

    “I was wondering,” Harry Potter said calmly, voice carrying impossible authority, “why my name was being taken in vain.”

    Everyone froze.

    Theo stood immediately, fist to chest. “Heritor.”

    Harry inclined his head slightly. “At ease.”

    Pansy’s knees nearly buckled.

    And far above them, the castle itself seemed to breathe, ancient magic stirring, as if acknowledging its true master """
    
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
