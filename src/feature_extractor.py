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
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    def extract_all_features(self, text: str, return_dict: bool = False):
        """Extract all 57 linguistic features"""

        # Parse text
        doc = self.nlp(text)

        # Extract feature groups
        syntactic_features_dict = self._extract_syntactic_templates(doc, return_dict=True)
        ngram_features_dict = self._extract_ngram_diversity(text, return_dict=True)
        psycho_features_dict = self._extract_psycholinguistic(text, doc, return_dict=True)

        if return_dict:
            all_features_dict = {}
            all_features_dict.update(syntactic_features_dict)
            all_features_dict.update(ngram_features_dict)
            all_features_dict.update(psycho_features_dict)
            return all_features_dict
        else:
            # Concatenate all features into a numpy array
            syntactic_features = np.array(list(syntactic_features_dict.values()))
            ngram_features = np.array(list(ngram_features_dict.values()))
            psycho_features = np.array(list(psycho_features_dict.values()))
            return np.concatenate([syntactic_features, ngram_features, psycho_features])

    def _extract_syntactic_templates(self, doc, return_dict: bool = False):
        """Extract 9 syntactic template features"""

        # Get POS tags
        pos_tags = [token.pos_ for token in doc]

        if len(pos_tags) < 4:
            if return_dict:
                return {
                    "repetition_rate": 0, "template_ttr": 0, "zipf_deviation": 0,
                    "avg_template_freq": 0, "max_template_freq": 0, "template_entropy": 0,
                    "unique_template_ratio_4": 0, "unique_template_ratio_6": 0, "unique_template_ratio_8": 0
                }
            return np.zeros(9)

        features_list = []
        feature_names = []

        # Extract templates of different lengths (4-8)
        all_templates = []
        for n in range(4, 9):
            templates = [
                tuple(pos_tags[i : i + n]) for i in range(len(pos_tags) - n + 1)
            ]
            all_templates.extend(templates)

        if not all_templates:
            if return_dict:
                return {
                    "repetition_rate": 0, "template_ttr": 0, "zipf_deviation": 0,
                    "avg_template_freq": 0, "max_template_freq": 0, "template_entropy": 0,
                    "unique_template_ratio_4": 0, "unique_template_ratio_6": 0, "unique_template_ratio_8": 0
                }
            return np.zeros(9)

        template_counts = Counter(all_templates)

        # Feature 1: Template repetition rate
        repeated_templates = sum(1 for count in template_counts.values() if count > 2)
        repetition_rate = (
            repeated_templates / len(template_counts) if template_counts else 0
        )
        features_list.append(repetition_rate)
        feature_names.append("repetition_rate")

        # Feature 2: Template Type-Token Ratio
        template_ttr = len(template_counts) / len(all_templates) if all_templates else 0
        features_list.append(template_ttr)
        feature_names.append("template_ttr")

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
        features_list.append(zipf_deviation)
        feature_names.append("zipf_deviation")

        # Additional template features (for total of 9)
        # Average template frequency
        avg_freq = np.mean(list(template_counts.values())) if template_counts else 0
        features_list.append(avg_freq)
        feature_names.append("avg_template_freq")

        # Max template frequency
        max_freq = max(template_counts.values()) if template_counts else 0
        features_list.append(max_freq)
        feature_names.append("max_template_freq")

        # Template entropy
        if template_counts:
            freqs = np.array(list(template_counts.values()))
            probs = freqs / freqs.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            entropy = 0
        features_list.append(entropy)
        feature_names.append("template_entropy")

        # Unique template ratio for different lengths
        for n in [4, 6, 8]:
            n_templates = [
                tuple(pos_tags[i : i + n]) for i in range(len(pos_tags) - n + 1)
            ]
            if n_templates:
                unique_ratio = len(set(n_templates)) / len(n_templates)
            else:
                unique_ratio = 0
            features_list.append(unique_ratio)
            feature_names.append(f"unique_template_ratio_{n}")

        if return_dict:
            return dict(zip(feature_names, features_list))
        return np.array(features_list)
    def _extract_ngram_diversity(self, text: str, return_dict: bool = False):
        """Extract 14 n-gram diversity features"""

        words = text.split()

        if len(words) < 2:
            if return_dict:
                return {f"ngram_ttr_{n}": 0 for n in range(2,8)} | {f"hapax_rate_{n}": 0 for n in range(2,8)} | {"mtld_score": 0, "basic_ttr": 0}
            return np.zeros(14)

        features_list = []
        feature_names = []

        # Word-level n-grams (2-7)
        for n in range(2, 8):
            if len(words) >= n:
                ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
                ngram_counts = Counter(ngrams)

                # Type-Token Ratio
                ttr = len(ngram_counts) / len(ngrams) if ngrams else 0
                features_list.append(ttr)
                feature_names.append(f"ngram_ttr_{n}")

                # Hapax legomena rate
                hapax_count = sum(1 for count in ngram_counts.values() if count == 1)
                hapax_rate = hapax_count / len(ngram_counts) if ngram_counts else 0
                features_list.append(hapax_rate)
                feature_names.append(f"hapax_rate_{n}")
            else:
                features_list.extend([0, 0])
                feature_names.extend([f"ngram_ttr_{n}", f"hapax_rate_{n}"])

        # MTLD (Measure of Textual Lexical Diversity)
        try:
            mtld_score = ld.mtld(words)
        except Exception: # Catching generic Exception for robustness
            mtld_score = 0
        features_list.append(mtld_score)
        feature_names.append("mtld_score")

        # Basic TTR
        basic_ttr = len(set(words)) / len(words) if words else 0
        features_list.append(basic_ttr)
        feature_names.append("basic_ttr")

        if return_dict:
            return dict(zip(feature_names, features_list))
        return np.array(features_list)
    def _extract_psycholinguistic(self, text: str, doc, return_dict: bool = False):
        """Extract 34 psycholinguistic features"""

        features_list = []
        feature_names = []
        words = text.split()
        sentences = list(doc.sents)

        # Lexical diversity (4 features)
        unique_words = len(set(words))
        features_list.append(unique_words)
        feature_names.append("unique_words")

        ttr = len(set(words)) / len(words) if words else 0
        features_list.append(ttr)
        feature_names.append("ttr")

        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        features_list.append(avg_word_length)
        feature_names.append("avg_word_length")

        long_word_ratio = (
            sum(1 for w in words if len(w) > 6) / len(words) if words else 0
        )
        features_list.append(long_word_ratio)
        feature_names.append("long_word_ratio")

        # Syntactic complexity (8 features)
        sentence_lengths = [len(list(sent)) for sent in sentences]
        features_list.append(np.mean(sentence_lengths) if sentence_lengths else 0)
        feature_names.append("avg_sentence_length")
        features_list.append(np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0)
        feature_names.append("std_sentence_length")
        features_list.append(min(sentence_lengths) if sentence_lengths else 0)
        feature_names.append("min_sentence_length")
        features_list.append(max(sentence_lengths) if sentence_lengths else 0)
        feature_names.append("max_sentence_length")

        # Complex sentences (with subordinate clauses)
        complex_count = sum(
            1
            for sent in sentences
            if any(token.dep_ in ["mark", "advcl"] for token in sent)
        )
        features_list.append(complex_count / len(sentences) if sentences else 0)
        feature_names.append("complex_sentence_ratio")

        # Punctuation patterns (3 types)
        comma_ratio = text.count(",") / len(words) if words else 0
        semicolon_ratio = text.count(";") / len(words) if words else 0
        emdash_ratio = text.count("—") / len(words) if words else 0
        features_list.extend([comma_ratio, semicolon_ratio, emdash_ratio])
        feature_names.extend(["comma_ratio", "semicolon_ratio", "emdash_ratio"])

        # Readability (2 features)
        try:
            flesch_score = flesch_reading_ease(text)
        except Exception:
            flesch_score = 0
        features_list.append(flesch_score)
        feature_names.append("flesch_reading_ease")

        try:
            fog_score = gunning_fog(text)
        except Exception:
            fog_score = 0
        features_list.append(fog_score)
        feature_names.append("gunning_fog")

        # POS distribution (6 features)
        pos_counts = Counter([token.pos_ for token in doc])
        total_tokens = len(doc)

        for pos in ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET"]:
            ratio = pos_counts.get(pos, 0) / total_tokens if total_tokens > 0 else 0
            features_list.append(ratio)
            feature_names.append(f"pos_ratio_{pos.lower()}")

        # Named entities (3 features)
        # Re-enable NER for this
        doc_with_ner = self.nlp(text)
        entities = [ent for ent in doc_with_ner.ents]

        entity_density = len(entities) / len(words) if words else 0
        features_list.append(entity_density)
        feature_names.append("entity_density")

        # Person entities
        person_entities = sum(1 for ent in entities if ent.label_ == "PERSON")
        features_list.append(person_entities / len(words) if words else 0)
        feature_names.append("person_entity_ratio")

        # First person pronouns
        first_person = sum(
            1
            for token in doc
            if token.text.lower()
            in ["i", "me", "my", "mine", "we", "us", "our", "ours"]
        )
        features_list.append(first_person / len(words) if words else 0)
        feature_names.append("first_person_pronoun_ratio")

        # Active vs passive voice (2 features)
        passive_count = sum(1 for token in doc if token.dep_ == "auxpass")
        passive_ratio = passive_count / len(list(doc.sents)) if list(doc.sents) else 0
        features_list.append(passive_ratio)
        feature_names.append("passive_voice_ratio")

        active_ratio = 1 - passive_ratio
        features_list.append(active_ratio)
        feature_names.append("active_voice_ratio")

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
        features_list.append(marker_count / len(words) if words else 0)
        feature_names.append("discourse_marker_ratio")

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
        features_list.append(transition_count / len(words) if words else 0)
        feature_names.append("transition_word_ratio")

        # Stopword ratio (1 feature)
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
        stopword_count = sum(1 for word in words if word.lower() in stop_words)
        features_list.append(stopword_count / len(words) if words else 0)
        feature_names.append("stopword_ratio")

        # Question marks and exclamation points (2 features)
        features_list.append(text.count("?") / len(sentences) if sentences else 0)
        feature_names.append("question_mark_ratio")
        features_list.append(text.count("!") / len(sentences) if sentences else 0)
        feature_names.append("exclamation_mark_ratio")

        # Parenthetical phrases (1 feature)
        paren_count = text.count("(") + text.count("[")
        features_list.append(paren_count / len(sentences) if sentences else 0)
        feature_names.append("parenthetical_phrase_ratio")

        if return_dict:
            return dict(zip(feature_names, features_list))
        return np.array(features_list)



# Usage
extractor = LinguisticFeatureExtractor()
sample_text = "This is a sample text for feature extraction."
features = extractor.extract_all_features(sample_text)
print(f"Extracted {len(features)} features")
