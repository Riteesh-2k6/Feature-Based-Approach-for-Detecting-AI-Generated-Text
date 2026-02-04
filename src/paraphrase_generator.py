# paraphrase_generator.py
from transformers import pipeline
# Add at top:
import pandas as pd

def generate_paraphrased_dataset(df, num_samples=1000):
    """Generate paraphrased versions of AI-generated texts"""
    
    paraphraser = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")
    
    ai_samples = df[df["label"] == 1].sample(num_samples)
    
    paraphrased_data = []
    for idx, row in ai_samples.iterrows():
        # Fixed: Access result correctly
        result = paraphraser(row["text"], max_length=512, num_return_sequences=1)
        paraphrase = result[0]["generated_text"]  # Fixed
        
        paraphrased_data.append({
            "text": paraphrase,
            "label": 1,
            "is_paraphrased": True,
        })
    
    return pd.DataFrame(paraphrased_data)


# Usage (optional - for adversarial testing)
# paraphrased_test = generate_paraphrased_dataset(test_df, num_samples=500)
# paraphrased_test.to_csv("data/test_paraphrased.csv", index=False)
