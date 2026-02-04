# Linguistic Features Documentation (54 Features)

This document describes the 54 linguistic features extracted by the `LinguisticFeatureExtractor` used in the Hybrid AI Text Detector. These features are grouped into three main categories: Syntactic Templates, N-gram Diversity, and Psycholinguistic/Stylometric features.

---

## 1. Syntactic Template Features (9 Features)
These features analyze the structure of Part-of-Speech (POS) sequences (templates) of lengths 4 to 8. They capture the structural "skeleton" of the writing.

| Feature Name | Description |
| :--- | :--- |
| **repetition_rate** | The proportion of POS templates (sequences of POS tags) that appear more than twice. High repetition suggests a formulaic or repetitive sentence structure. |
| **template_ttr** | Template Type-Token Ratio. The ratio of unique POS templates to total templates. Measures the overall structural variety of the text. |
| **zipf_deviation** | Measures how much the frequency distribution of POS templates deviates from Zipf's Law. AI-generated text often follows Zipf's Law more strictly than human text. |
| **avg_template_freq** | The average number of times each POS template is reused in the text. |
| **max_template_freq** | The frequency of the most common POS template used in the text. |
| **template_entropy** | Shannon entropy of the POS template distribution. Higher entropy indicates more unpredictable and diverse sentence structures. |
| **unique_template_ratio_4** | The ratio of unique POS sequences of length 4 to the total number of length-4 sequences. |
| **unique_template_ratio_6** | The ratio of unique POS sequences of length 6. |
| **unique_template_ratio_8** | The ratio of unique POS sequences of length 8. |

---

## 2. N-gram Diversity Features (14 Features)
These features analyze word-level patterns (n-grams) to measure vocabulary richness and phrasing repetition.

| Feature Name | Description |
| :--- | :--- |
| **ngram_ttr_2** to **7** | Type-Token Ratio for n-grams of lengths 2 through 7. Measures how often the author repeats specific phrases of various lengths. |
| **hapax_rate_2** to **7** | The ratio of n-grams (lengths 2-7) that appear exactly once. A high "Hapax Legomena" rate indicates creative and non-repetitive phrasing. |
| **mtld_score** | Measure of Textual Lexical Diversity. A robust measure of vocabulary richness that is less sensitive to text length than standard TTR. |
| **basic_ttr** | Simple Type-Token Ratio for individual words. The ratio of unique words to total words. |

---

## 3. Psycholinguistic & Stylometric Features (31 Features)
These features capture specific writing styles, complexity levels, and grammatical preferences.

### Lexical & Readability
| Feature Name | Description |
| :--- | :--- |
| **unique_words** | Total count of unique words in the text. |
| **ttr** | Redundant Type-Token Ratio (Word level). |
| **avg_word_length** | Average number of characters per word. |
| **long_word_ratio** | Proportion of words longer than 6 characters. Indicates vocabulary sophistication. |
| **flesch_reading_ease** | Standard readability score (0-100). Higher scores are easier to read. |
| **gunning_fog** | Readability score indicating the years of formal education needed to understand the text. |

### Sentence Structure
| Feature Name | Description |
| :--- | :--- |
| **avg_sentence_length** | Average number of words per sentence. |
| **std_sentence_length** | Standard deviation of sentence lengths. Humans typically vary sentence lengths more than AI. |
| **min_sentence_length** | Length of the shortest sentence. |
| **max_sentence_length** | Length of the longest sentence. |
| **complex_sentence_ratio** | Ratio of sentences containing subordinate clauses (complex syntax). |

### Grammar & POS Distribution
| Feature Name | Description |
| :--- | :--- |
| **pos_ratio_noun** | Frequency of nouns relative to total tokens. |
| **pos_ratio_verb** | Frequency of verbs. |
| **pos_ratio_adj** | Frequency of adjectives. |
| **pos_ratio_adv** | Frequency of adverbs. |
| **pos_ratio_pron** | Frequency of pronouns. |
| **pos_ratio_det** | Frequency of determiners (e.g., the, a, this). |
| **passive_voice_ratio** | Proportion of sentences using passive voice constructions. |
| **active_voice_ratio** | Proportion of sentences using active voice (1 - passive_voice_ratio). |
| **stopword_ratio** | Ratio of common function words (the, is, in) to total words. |

### Semantic & Discourse
| Feature Name | Description |
| :--- | :--- |
| **entity_density** | Frequency of Named Entities (People, Places, Organizations) per word. |
| **person_entity_ratio** | Frequency of Person entities specifically. |
| **first_person_pronoun_ratio** | Frequency of first-person pronouns (I, me, we, etc.). Humans often use these more in personal writing. |
| **discourse_marker_ratio** | Frequency of logical connectors (however, therefore, consequently). AI often overuses these. |
| **transition_word_ratio** | Frequency of sequence words (first, finally, additionally). |

### Punctuation & Style
| Feature Name | Description |
| :--- | :--- |
| **comma_ratio** | Frequency of commas per word. |
| **semicolon_ratio** | Frequency of semicolons per word. |
| **emdash_ratio** | Frequency of em-dashes (—) per word. |
| **question_mark_ratio** | Average number of question marks per sentence. |
| **exclamation_mark_ratio** | Average number of exclamation marks per sentence. |
| **parenthetical_phrase_ratio** | Frequency of parenthetical notes using brackets `()` or `[]`. |
