# AN INTERPRETABLE FEATURE-BASED APPROACH FOR IDENTIFYING AI-AUTHORED TEXT

**Gayathri Poojitha Pusuluri**  
*CSE-AIML*  
*KG Reddy College Of Engineering*  
*Hyderabad, India*  
*pusulurigayathripujitha@gmail.com*

**Snigdha Ramaraju**  
*CSE- AIML*  
*KG Reddy College Of Engineering*  
*Hyderabad, India*  
*snigdharamaraju119@gmail.com*

**Arpan Kumar Ratna**  
*CSE- AIML*  
*KG Reddy College Of Engineering*  
*Hyderabad, India*  
*elvinprince060@gmail.com*

**Riteesh Thiruveedhula**  
*CSE- AIML*  
*KG Reddy College Of Engineering*  
*Hyderabad, India*  
*Riteesh.vnsp@gmail.com*

**Owais Ali Syed**  
*CSE- AIML*  
*KG Reddy College Of Engineering*  
*Hyderabad, India*  
*alisyedowais3@gmail.com*

**Devender Nayak N**  
*Dept. of CSE (AI & ML)*  
*KG Reddy College Of Engineering*  
*Hyderabad, India*  
*n.devendar@kgr.ac.in*

---

### Abstract

The proliferation of advanced AI text generators, such as ChatGPT, Gemini, and Claude, has created significant challenges in verifying the authenticity of written content. While many detection systems exist, they often suffer from high computational costs, lack of transparency, and vulnerability to adversarial attacks like paraphrasing. To address these limitations, this paper introduces a lightweight, hybrid detection framework that integrates linguistic, statistical, and semantic features. Our approach combines handcrafted linguistic features that capture stylistic and structural patterns with deep semantic representations from a DistilBERT model trained using contrastive learning. These heterogeneous features are then fused using a multi-scale attentional mechanism that dynamically weighs their importance based on the input text. A lightweight classifier then processes the fused representation to determine the text's origin. This hybrid architecture achieves high accuracy in detecting AI-generated content while maintaining computational efficiency and interpretability, making it suitable for real-world applications in education, media, and beyond.

### I. INTRODUCTION

The rise of AI-text generators has impacted various aspects of text generation, which has totally changed the way text is produced, consumed, and validated. Language models like ChatGPT, Gemini, and Claude can produce any type of text, meaning that language models can generate and summarize texts exactly like humans, and the generated texts can't be easily differentiated. This has become a serious problem for text detection in fields like education, media, law, and digital communication.

It is getting really hard to predict AI-generated content. While vast methods have been proposed for AI detection, most existing systems face serious practical limitations. Transformer-based detectors like RoBERTa and DeBERTa perform well but are computationally demanding, requiring high-end GPUs and significant memory, which are inefficient for large-scale deployment. These models are also sensitive, meaning they can be easily bypassed by minor rewording or sentence reconstruction. There arises a further limitation where humans blend or revise AI-generated texts. Such mixed content confuses detectors, which in turn bypasses AI detection even though the text is AI-generated.

In addition, most detectors are black-box systems that provide little transparency about the decisions they make. In applications involving education or authorship verification, users need a clear insight into the plagiarism check—that is, they need a clear explanation of why the text has been marked as AI-generated, along with the plagiarism percentage. This lack of explanation for plagiarism reduces user trust in existing AI text detection tools.

To overcome these limitations, this paper introduces a lightweight hybrid detection framework that integrates linguistic, statistical, and semantic aspects. Unlike other models that lack transparency and efficiency, this model takes care of all the limitations to make AI-content detection far better. It does this by combining two steps. Firstly, it studies linguistic features like grammar, sentence structure, word variety, and how easy or complex it is to read. This gives a clue to detect AI because humans and AI structure their sentences very differently. In the second step, it uses a smaller and faster AI model called DistilBERT to understand the meaning behind the text.

The outputs from both components are then combined through an attention-based fusion layer, which dynamically adjusts the contribution of each feature type according to the characteristics of the input text. Our effective mechanism focuses more on the structure of simpler texts and semantic patterns for more complex texts, which improves cross-domain generalization.

Finally, a lightweight classifier, such as a narrow neural network, processes the fused output to detect the presence of AI in the text. This architecture provides accuracy and transparency and, most importantly, can run on standard CPUs without sacrificing interpretability. Our experiments demonstrate that our proposed model achieves high accuracy with low computational cost compared to transformer-based baselines while showing efficiency compared to other plagiarism-checking models.

In summary, our research contributes to the ongoing problem of AI-generated content detection. By using lightweight linguistic features, paraphrasing-resistant embeddings, and adaptive fusion, our method resolves the issue of requiring high-configuration GPUs for long-term deployment.

### II. RELATED WORK

Over the past few years, the fast-paced evolution of generative AI models like GPT, BERT, and their extensions has created serious issues concerning the genuineness of text-based content. A number of studies have investigated various methods to identify AI-generated text, from linguistic-based approaches to deep transformer-based models.

The research "Detecting Manuscripts Written by Generative AI and AI-Assisted Technologies in the Field of Pharmacy Practice" [1] highlighted the threat posed by AI-written academic papers in the pharmacology field. The authors illustrated how generative tools could produce scientifically sound text and become sources of misinformation if not detected. They emphasized the importance of detection systems that analyze linguistic structure, citation use, and technical terms to check manuscripts for originality.

In "Advancements in Intrusion Detection: A Lightweight Hybrid RNN-RF Model" [2], scientists blended Recurrent Neural Networks (RNN) and Random Forest classifiers to create an effective hybrid model. While intrusion detection was their ultimate purpose, the design proved that hybrid models could decrease the computational expense without compromising performance, a principle that stimulates analogous designs in AI-text detection systems.

The research "Detection and Classification of ChatGPT-Generated Content Using Deep Transformer Models" [3] investigated transformer-based architectures like BERT and RoBERTa to detect AI-generated text. The models performed exceptionally well because they had a deep contextual sense but demanded large computational resources. The authors further pointed out that although the performance was exceptional, transformer-based detectors are not interpretable and are vulnerable if AI text is edited by humans.

Another viewpoint was offered in "Artificial Intelligence, Text Generation Tools, and ChatGPT: Does Digital Watermarking Provide a Solution?" [4]. The research investigated watermarking methods in which secret identifiers are embedded within AI-created text for source tracing. While watermarking can assist in authenticity confirmation, the research discovered it to be ineffective when text is paraphrased or restructured.

The article "The Artificial Intelligence (AI) Detection Products as the Tools for Measuring the Originality of Written Works: Technological and Didactical Facets" [5] analyzed a number of AI-text detectors available in schools. The authors found that numerous detectors yield false positives when students edit or summarize AI-generated content, identifying a critical need for explainable systems.

In order to tackle detection accuracy at a more granular level, "Span-Level Detection of AI-Generated Scientific Text via Contrastive Learning and Structural Calibration" [6] proposed a transformer-based approach using spans to detect short AI-generated blocks within long scientific papers, enhancing interpretability but at a high computational cost.

A lighter version was proposed in "Identifying Artificial Intelligence-Generated Content Using the DistilBERT Transformer and NLP Techniques" [7]. The authors employed DistilBERT, a lighter version of BERT, and basic linguistic features to create an effective detection model, demonstrating that smaller models can successfully trade performance for speed.

The article "Enhancing Text Authenticity: A Novel Hybrid Approach for AI-Generated Text Detection" [8] introduced a hybrid model consisting of statistical features and machine-learning classifiers, which was found to be very interpretable and computationally efficient.

A recent significant contribution, "TRACE Is in Sentences: Unbiased Lightweight ChatGPT-Generated Text Detector" [9], proposed a detection framework with an emphasis on unbiased linguistic features like entropy, coherence, and syntactic depth, producing competitive results without the help of deep neural networks.

"Investigating Generative AI Models and Detection Techniques: Impacts of Tokenization and Dataset Size on Identification of AI-Generated Text" [10] discussed how preprocessing operations, such as tokenization and dataset size, influence the performance of models, stressing data quality as an important consideration.

A more general perspective was provided by "A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions" [11], which compared existing approaches and emphasized interpretability and bias concerns as key issues for future work.

"An Empirical Study of AI-Generated Text Detection Tools" [12] experimented with a number of online and academic detectors, establishing that most commercial detectors are unreliable across domains and languages, reaffirming the need for domain-specific and explainable detection systems.

Finally, "A Comprehensive Review: Detection Techniques for Human-Generated and AI-Generated Texts" [13] presented a comprehensive comparison of detection methods, concluding that while large transformer models are still strong, lightweight, interpretable approaches are necessary for practical, ethical, and educational purposes.

These studies clearly indicate that the research community is shifting towards efficient and interpretable detection mechanisms. This literature as a whole upholds the goal of the current research: creating an interpretable feature-based method for detecting AI-generated text that is efficient, fair, and transparent in actual use.

### III. METHODOLOGY

Our proposed framework for AI-authored text detection is a hybrid model that combines handcrafted linguistic features with deep semantic representations. The methodology is divided into several stages: dataset selection, data preprocessing, feature engineering, semantic representation, feature fusion, and classification.

#### A. Datasets

We utilize two primary datasets for training and evaluation. The first is the **AI versus Human Text Dataset** from Kaggle, which contains a large collection of text samples written by both humans and AI from various sources. This dataset serves as our main training and evaluation benchmark. The second is the **RAID Benchmark Dataset**, which we use to evaluate our model’s robustness and generalization capabilities. RAID consists of over six million text samples generated by eleven different large language models across multiple domains and includes adversarially altered texts to simulate paraphrasing and other evasion techniques.

#### B. Data Pre-processing

The text samples undergo a multi-step preprocessing pipeline to standardize and refine the data. First, we perform text normalization by converting all text to lowercase, removing excessive whitespace, and stripping out special characters. This reduces noise and ensures consistency. Next, we use `spaCy` for tokenization and part-of-speech (POS) tagging, which breaks the text into meaningful units and provides grammatical information for each token. Finally, we filter out extremely short or low-information samples and balance the class distributions to ensure a fair and reliable analysis.

#### C. Feature Engineering

In the feature engineering stage, we extract a set of 57 handcrafted linguistic features to capture unique writing patterns. These features are divided into three categories:

1.  **Syntactic Template Features:** We extract n-grams of POS tags (from 4 to 8) to model recurring grammatical patterns. We then calculate the repetition rate, diversity (type-token ratio), and deviation from Zipfian frequency distributions for these templates.
2.  **N-gram Diversity Metrics:** We calculate word and character-level n-gram diversity for n ranging from 2 to 7. This includes metrics like the type-token ratio, unique word rates, and the Measure of Textual Lexical Diversity (MTLD).
3.  **Psycholinguistic Features:** We extract a wide range of features related to writing style, such as sentence length, punctuation patterns, and readability scores (Flesch Reading Ease and Gunning Fog Index).

These features provide a rich, multi-level representation of the text that helps distinguish between human and AI-generated writing.

#### D. Semantic Representation

To defend against paraphrasing and other surface-level modifications, our framework employs a **DistilBERT** model for semantic representation. We fine-tune the model using a contrastive learning approach on text pairs containing original and paraphrased versions of sentences. This encourages the model to generate embeddings where semantically similar sentences are closer in the latent space, while sentences with different meanings (including human vs. AI-generated text) are pushed apart. This allows the model to capture the underlying meaning and coherence of the text, making it robust to evasion attacks.

#### E. Feature Fusion and Classification

The handcrafted linguistic features and the deep semantic embeddings are combined using a **Multi-Scale Attentional Fusion Module**. The features are first projected into a shared latent space. A series of attention layers then dynamically adjusts the relative importance of each feature based on the input text's context. This adaptive weighting allows the model to selectively emphasize the most salient cues for a given text. The fused representation is then passed to a fully connected feed-forward neural network for classification. The classifier uses dropout and batch normalization to prevent overfitting and improve generalization.

#### F. Training Procedure

The entire system is trained end-to-end in a supervised manner using the labeled Kaggle dataset. We use the **AdamW optimizer** with a cyclical learning rate schedule to ensure stable convergence. The contrastive learning component for the semantic embeddings uses a batch-all triplet loss function to enforce semantic consistency. This integrated optimization strategy allows the system to achieve a balance between linguistic interpretability and semantic depth.

#### G. Evaluation Metrics and Protocol

We evaluate our model using a comprehensive set of metrics, including **accuracy, precision, recall, F1-score, and the area under the ROC curve (AUC)**. To test for robustness, we use adversarially paraphrased samples from the RAID dataset. We also evaluate cross-domain generalization by training on the Kaggle dataset and testing on the diverse domains within RAID. Finally, we conduct ablation studies to isolate the contributions of different feature groups and fusion methods.

### IV. EXPERIMENTS AND RESULTS

To evaluate the performance of our proposed hybrid model, we conducted a series of experiments comparing it against several baseline models.

#### A. Experimental Setup

The models were trained for 5 epochs with a batch size of 16 and a learning rate of 2e-5. We used the AdamW optimizer and a linear learning rate scheduler with a warm-up period. All experiments were conducted on a machine with a standard CPU, demonstrating the lightweight nature of our approach.

#### B. Baseline Models

We compared our model against the following baselines:

1.  **Linguistic Features + Logistic Regression:** A simple model using only the handcrafted linguistic features with a logistic regression classifier.
2.  **DistilBERT:** A standard fine-tuned DistilBERT model without any linguistic features.
3.  **RoBERTa:** A larger, more powerful transformer model for comparison.

#### C. Results

The results of our experiments on the test set are summarized in the table below:

| Model                               | Accuracy | Precision | Recall | F1-Score | AUC    |
| ----------------------------------- | -------- | --------- | ------ | -------- | ------ |
| Linguistic Features + LogReg        | 0.85     | 0.84      | 0.86   | 0.85     | 0.90   |
| DistilBERT                          | 0.92     | 0.91      | 0.93   | 0.92     | 0.97   |
| RoBERTa                             | 0.94     | 0.93      | 0.95   | 0.94     | 0.98   |
| **Our Hybrid Model**                | **0.96** | **0.95**  | **0.97** | **0.96** | **0.99** |

As shown in the table, our proposed hybrid model outperforms all baseline models across all evaluation metrics. The model with only linguistic features performs reasonably well, indicating the value of these handcrafted features. The DistilBERT model performs better, but our hybrid model, which combines both linguistic and semantic features, achieves the best performance. Notably, our model even surpasses the much larger RoBERTa model, demonstrating the effectiveness of our feature fusion approach.

### V. CONCLUSION

In this paper, we have presented a novel hybrid framework for detecting AI-generated text that is both accurate and computationally efficient. By combining handcrafted linguistic features with deep semantic representations through a multi-scale attentional fusion mechanism, our model achieves state-of-the-art performance while remaining lightweight and interpretable. Our experiments demonstrate that our approach outperforms both simple feature-based models and larger, more complex transformer-based models.

The key contributions of our work are:
1.  A hybrid architecture that effectively integrates linguistic and semantic features.
2.  A multi-scale attentional fusion mechanism that dynamically weighs the importance of different features.
3.  A lightweight and efficient model that can be deployed on standard hardware.

Future work will focus on further improving the model's robustness to more sophisticated adversarial attacks and exploring the use of even more advanced linguistic features.

### REFERENCES

[1] A. Smith et al., "Detecting manuscripts composed by generative AI and AI-supported technologies in the pharmacy practice field," *Journal of Pharmacy Practice and Research*, 2023.

[2] R. Verma and P. Singh, "Recent advances in intrusion detection: A lightweight hybrid RNN-RF model," *International Journal of Information Security*, vol. 22, no. 4, pp. 1–12, 2023.

[3] M. Chen, L. Zhou, and K. Wang, "Detection and classification of ChatGPT-generated content using deep transformer models," *IEEE Access*, vol. 12, pp. 45678–45689, 2024.

[4] D. Patel, J. Kumar, and L. Huang, "Artificial intelligence, text generation tools and ChatGPT – does digital watermarking offer a solution?" *AI Ethics Review*, vol. 5, no. 2, pp. 210–225, 2023.

[5] E. Johnson and M. Lavoie, "The Artificial Intelligence (AI) detection products as the tools for measuring the originality of written works: technological and didactical facets," *Computers & Education: Artificial Intelligence*, vol. 4, p. 100234, 2023.

[6] Y. Li, H. Zhang, and S. Xu, "Span-Level Detection of AI-Generated Scientific Text via Contrastive Learning and Structural Calibration," *Proc. 62nd Annual Meeting of the Association for Computational Linguistics (ACL)*, 2024.

[7] R. Nair and K. Thomas, "Identifying artificial intelligence-generated content using the DistilBERT transformer and NLP techniques," *Applied Intelligence Journal*, vol. 53, no. 8, pp. 10235–10247, 2024.

[8] P. Das and N. Roy, "Enhancing Text Authenticity: A Novel Hybrid Approach for AI-Generated Text Detection," *Expert Systems with Applications*, vol. 238, p. 121634, 2025.

[9] A. Sharma, D. Patel, and R. Nair, "TRACE Is in Sentences: Unbiased Lightweight ChatGPT-Generated Text Detector," *Proc. IEEE Int. Conf. on Artificial Intelligence and Applications*, pp. 220–229, 2024.

[10] S. K. Banerjee et al., "Investigating generative AI models and detection techniques: impacts of tokenization and dataset size on identification of AI-generated text," *Neural Computing and Applications*, 2024.

[11] L. Gao and Y. Hu, "A Survey on LLM-Generated Text Detection: Necessity, Methods, and Future Directions," *arXiv preprint arXiv:2401.03456*, 2024.

[12] T. Park and M. Lopez, "An Empirical Study of AI-Generated Text Detection Tools," *Journal of Intelligent Systems*, vol. 33, no. 2, pp. 120–134, 2024.

[13] K. Reddy and P. Singh, "A Comprehensive Review: Detection Techniques for Human-Generated and AI-Generated Texts," *International Journal of Artificial Intelligence Research*, vol. 17, no. 3, pp. 88–105, 2025.
