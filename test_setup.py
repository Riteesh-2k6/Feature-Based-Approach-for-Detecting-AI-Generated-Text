# test_setup.py
print("Testing installation...")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers: {e}")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print(f"✓ spaCy: {spacy.__version__}")
    print(f"✓ spaCy model loaded")
except Exception as e:
    print(f"✗ spaCy: {e}")

try:
    import nltk
    print(f"✓ NLTK: {nltk.__version__}")
except Exception as e:
    print(f"✗ NLTK: {e}")

try:
    import sklearn
    print(f"✓ scikit-learn: {sklearn.__version__}")
except Exception as e:
    print(f"✗ scikit-learn: {e}")

try:
    import pandas
    print(f"✓ pandas: {pandas.__version__}")
except Exception as e:
    print(f"✗ pandas: {e}")

print("\n✓ All dependencies installed successfully!")
