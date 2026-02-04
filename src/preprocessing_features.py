# feature_extractor.py
import spacy
import numpy as np
from collections import Counter
from textstat import flesch_reading_ease, gunning_fog
from lexical_diversity import lex_div as ld
import nltk
from nltk.corpus import stopwords

# Add download in __init__:


class LinguisticFeatureExtractor:
    def __init__(self):
        # self.nlp = spacy.load("en_core_web_sm", disable=["ner"]) # REMOVED
        
        # We load a separate, NER-enabled model just for the NER task
        # This fixes a bug where NER was being run with a disabled model
        try:
            self.nlp_ner = spacy.load("en_core_web_sm")
        except IOError:
            print("Error: 'en_core_web_sm' model not found. Please run:")
            print("python -m spacy download en_core_web_sm")
            raise

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    def extract_all_features(self, doc, text: str) -> np.ndarray:
        """Extract all 57 linguistic features (from pre-parsed doc)"""

        # Parse text
        # doc = self.nlp(text) # REMOVED - doc is now passed in

        # Extract feature groups
        syntactic_features = self._extract_syntactic_templates(doc)
        ngram_features = self._extract_ngram_diversity(text)
        psycho_features = self._extract_psycholinguistic(text, doc)

        # Concatenate all features
        all_features = np.concatenate(
            [syntactic_features, ngram_features, psycho_features]
        )

        return all_features

    def _extract_syntactic_templates(self, doc) -> np.ndarray:
        """Extract 9 syntactic template features"""

        # Get POS tags
        pos_tags = [token.pos_ for token in doc]

        if len(pos_tags) < 4:
            return np.zeros(9)

        features = []

        # Extract templates of different lengths (4-8)
        all_templates = []
        for n in range(4, 9):
            templates = [
                tuple(pos_tags[i : i + n]) for i in range(len(pos_tags) - n + 1)
            ]
            all_templates.extend(templates)

        if not all_templates:
            return np.zeros(9)

        template_counts = Counter(all_templates)

        # Feature 1: Template repetition rate
        repeated_templates = sum(1 for count in template_counts.values() if count > 2)
        repetition_rate = (
            repeated_templates / len(template_counts) if template_counts else 0
        )
        features.append(repetition_rate)

        # Feature 2: Template Type-Token Ratio
        template_ttr = len(template_counts) / len(all_templates) if all_templates else 0
        features.append(template_ttr)

        # Feature 3-9: Zipf deviation (simplified)
        # Calculate frequency distribution
        sorted_freqs = sorted(template_counts.values(), reverse=True)

        # Expected Zipf frequencies
        if sorted_freqs:
            zipf_expected = [
                sorted_freqs[0] / (r + 1) for r in range(len(sorted_freqs))
            ]
        else:
            zipf_expected = []

        # Calculate deviation
        if len(sorted_freqs) > 0:
            zipf_deviation = np.mean(
                np.abs(np.log1p(sorted_freqs) - np.log1p(zipf_expected))
            )
        else:
            zipf_deviation = 0
        features.append(zipf_deviation)

        # Additional template features (for total of 9)
        # Average template frequency
        avg_freq = np.mean(list(template_counts.values())) if template_counts else 0
        features.append(avg_freq)

        # Max template frequency
        max_freq = max(template_counts.values()) if template_counts else 0
        features.append(max_freq)

        # Template entropy
        if template_counts:
            freqs = np.array(list(template_counts.values()))
            probs = freqs / freqs.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            entropy = 0
        features.append(entropy)

        # Unique template ratio for different lengths
        for n in [4, 6, 8]:
            n_templates = [
                tuple(pos_tags[i : i + n]) for i in range(len(pos_tags) - n + 1)
            ]
            if n_templates:
                unique_ratio = len(set(n_templates)) / len(n_templates)
            else:
                unique_ratio = 0
            features.append(unique_ratio)

        return np.array(features[:9])

    def _extract_ngram_diversity(self, text: str) -> np.ndarray:
        """Extract 14 n-gram diversity features"""

        words = text.split()

        if len(words) < 2:
            return np.zeros(14)

        features = []

        # Word-level n-grams (2-7)
        for n in range(2, 8):
            if len(words) >= n:
                ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
                ngram_counts = Counter(ngrams)

                # Type-Token Ratio
                ttr = len(ngram_counts) / len(ngrams) if ngrams else 0
                features.append(ttr)

                # Hapax legomena rate
                hapax_count = sum(1 for count in ngram_counts.values() if count == 1)
                hapax_rate = hapax_count / len(ngram_counts) if ngram_counts else 0
                features.append(hapax_rate)
            else:
                features.extend([0, 0])

        # MTLD (Measure of Textual Lexical Diversity)
        try:
            mtld_score = ld.mtld(words)
        except Exception: # Fixed: Catch general exception
            mtld_score = 0
        features.append(mtld_score)

        # Basic TTR
        basic_ttr = len(set(words)) / len(words) if words else 0
        features.append(basic_ttr)

        return np.array(features[:14])

    def _extract_psycholinguistic(self, text: str, doc) -> np.ndarray:
        """Extract 34 psycholinguistic features"""

        features = []
        words = text.split()
        sentences = list(doc.sents)

        # Lexical diversity (4 features)
        unique_words = len(set(words))
        features.append(unique_words)

        ttr = len(set(words)) / len(words) if words else 0
        features.append(ttr)

        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        features.append(avg_word_length)

        long_word_ratio = (
            sum(1 for w in words if len(w) > 6) / len(words) if words else 0
        )
        features.append(long_word_ratio)

        # Syntactic complexity (8 features)
        sentence_lengths = [len(list(sent)) for sent in sentences]
        features.append(np.mean(sentence_lengths) if sentence_lengths else 0)
        features.append(np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0)
        features.append(min(sentence_lengths) if sentence_lengths else 0)
        features.append(max(sentence_lengths) if sentence_lengths else 0)

        # Complex sentences (with subordinate clauses)
        complex_count = sum(
            1
            for sent in sentences
            if any(token.dep_ in ["mark", "advcl"] for token in sent)
        )
        features.append(complex_count / len(sentences) if sentences else 0)

        # Punctuation patterns (3 types)
        comma_ratio = text.count(",") / len(words) if words else 0
        semicolon_ratio = text.count(";") / len(words) if words else 0
        emdash_ratio = text.count("—") / len(words) if words else 0
        features.extend([comma_ratio, semicolon_ratio, emdash_ratio])

        # Readability (2 features)
        try:
            flesch_score = flesch_reading_ease(text)
        except Exception: # Fixed: Catch general exception
            flesch_score = 0
        features.append(flesch_score)

        try:
            fog_score = gunning_fog(text)
        except Exception: # Fixed: Catch general exception
            fog_score = 0
        features.append(fog_score)

        # POS distribution (6 features)
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)

        for pos in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET"]:
            ratio = pos_counts.get(pos, 0) / total_tokens if total_tokens > 0 else 0
            features.append(ratio)

        # Named entities (3 features)
        # Use the separate, NER-enabled model
        doc_with_ner = self.nlp_ner(text) # FIXED BUG
        entities = [ent for ent in doc_with_ner.ents]

        entity_density = len(entities) / len(words) if words else 0
        features.append(entity_density)

        # Person entities
        person_entities = sum(1 for ent in entities if ent.label_ == "PERSON")
        features.append(person_entities / len(words) if words else 0)

        # First person pronouns
        first_person = sum(
            1
            for token in doc
            if token.text.lower()
            in ["i", "me", "my", "mine", "we", "us", "our", "ours"]
        )
        features.append(first_person / len(words) if words else 0)

        # Active vs passive voice (2 features)
        passive_count = sum(1 for token in doc if token.dep_ == "auxpass")
        passive_ratio = passive_count / len(list(doc.sents)) if list(doc.sents) else 0
        features.append(passive_ratio)

        active_ratio = 1 - passive_ratio
        features.append(active_ratio)

        # Discourse markers (2 features)
        discourse_markers = [
            "however",
            "therefore",
            "thus",
            "moreover",
            "furthermore",
            "consequently",
            "nevertheless",
        ]
        marker_count = sum(text.lower().count(marker) for marker in discourse_markers)
        features.append(marker_count / len(words) if words else 0)

        # Transitions
        transitions = [
            "first",
            "second",
            "finally",
            "next",
            "then",
            "additionally",
            "also",
        ]
        transition_count = sum(text.lower().count(trans) for trans in transitions)
        features.append(transition_count / len(words) if words else 0)

        # Stopword ratio (1 feature)
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
        stopword_count = sum(1 for word in words if word.lower() in stop_words)
        features.append(stopword_count / len(words) if words else 0)

        # Question marks and exclamation points (2 features)
        features.append(text.count("?") / len(sentences) if sentences else 0)
        features.append(text.count("!") / len(sentences) if sentences else 0)

        # Parenthetical phrases (1 feature)
        paren_count = text.count("(") + text.count("[")
        features.append(paren_count / len(sentences) if sentences else 0)

        return np.array(features[:34])


extractor = LinguisticFeatureExtractor()

if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm

    # Load the dataset in chunks
    chunk_size = 1000
    chunks = pd.read_csv('src/data/train.csv', chunksize=chunk_size)

    # Process each chunk
    all_features = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        # Extract features for each text in the chunk
        features = chunk['text'].apply(lambda text: extractor.extract_all_features(extractor.nlp_ner(text), text))
        all_features.extend(features)

    # Create a DataFrame with the features
    feature_df = pd.DataFrame(all_features)

    # Save the features to a new CSV file
    feature_df.to_csv('src/data/features.csv', index=False)

    print("Feature extraction complete. Features saved to 'src/data/features.csv'")

