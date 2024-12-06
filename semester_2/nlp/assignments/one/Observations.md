# Evaluation of N-Gram Language Models

- This Python code employs the NLTK library to preprocess text data, tokenize sentences, and build n-gram models of varying orders (unigram, bigram, and trigram). Training involves calculating probabilities of word sequences based on their frequencies in the training data. Evaluation is conducted by generating sentences and calculating their probabilities using both provided and curated test sets.

- Performance of n-gram language models (unigram, bigram, and trigram) is based on their ability to predict the likelihood of sentences. Each model was trained and tested on provided and curated datasets to assess its accuracy and effectiveness in capturing language patterns.

## Unigram Model

1. **Average Probabilities**:
   - **Provided Test Set**: `2.372106199716299e-18`
   - **Curated Test Set**: `6.044363807497101e-14`

2. **Observations**:
   - The unigram model shows extremely low probabilities for both test sets, indicating that the model struggles to accurately predict sequences of words. This is because the unigram model treats each word independently and does not consider the context provided by surrounding words. Hence, it fails to capture dependencies between words, resulting in poor predictive performance.

## Bigram Model

1. **Average Probabilities**:
   - **Provided Test Set**: `6.043652231489414e-14`
   - **Curated Test Set**: `1.5400434616281325e-09`

2. **Observations**:
   - The bigram model performs better than the unigram model, with higher probabilities for both test sets. This improvement is due to the fact that the bigram model considers pairs of consecutive words, thereby capturing some context and dependencies between adjacent words. This context sensitivity allows the model to make more informed predictions compared to the unigram model.

## Trigram Model

1. **Average Probabilities**:
   - **Provided Test Set**: `1.5400434616281325e-09`
   - **Curated Test Set**: `3.924338748920807e-05`

2. **Observations**:
   - The trigram model shows the highest probabilities among the three models for both test sets. This significant improvement is because the trigram model considers sequences of three consecutive words, which enhances its ability to capture complex patterns and dependencies in the text. By incorporating more context, the trigram model can more accurately predict the likelihood of entire sentences.

## General Observations

- **Model Performance**: The improvement from unigram to bigram to trigram models demonstrates the importance of context in language modeling. Models that consider more context (bigram and trigram) outperform the unigram model, which treats words in isolation.
  
- **Context Sensitivity**: The increasing performance from unigram to bigram to trigram models highlights the significance of context in language modeling. Trigram models, by considering three-word sequences, can capture more nuanced dependencies in natural language, leading to better predictive performance.

- **Data Sensitivity**: The different probabilities for the provided and curated test sets suggest that model performance can vary depending on the characteristics of the test data used. This underscores the importance of diverse and representative datasets in training and evaluating language models.

### Increasing test set variability

Increasing test set variability involves ensuring that the sentences in test set covers a diverse range of language patterns, contexts, and topics. To increase the variability of the test set, we need to consider the following strategies:

- **Diverse Sentence Sources**: Gather sentences from a wide range of sources such as news articles, books, scientific papers, social media posts, etc. Each source typically has its own style and vocabulary, contributing to variability.

- **Topic Diversity**: Ensure that the test set covers a variety of topics. This can include politics, technology, sports, entertainment, etc. Different topics often have distinct language usage and terminology.

- **Sentence Length and Complexity**: Include sentences of varying lengths and complexities. Some sentences may be short and simple, while others may be long and intricate, with clauses and complex structures.

- **Stylistic Variations**: Incorporate sentences with different stylistic elements, such as formal versus informal language, narrative versus descriptive styles, or technical versus conversational tones.

- **Grammar and Syntax**: Include sentences that vary in grammar and syntax. This can include different sentence structures (simple, compound, complex), varied use of punctuation, and syntactic complexity.

- **Rare Words and Phrases**: Introduce sentences that contain uncommon or domain-specific vocabulary. This challenges the model to handle unknown words and contexts gracefully.

- **Special Cases and Edge Scenarios**: Include sentences that test specific edge cases or scenarios, such as ambiguous meanings, idiomatic expressions, or linguistic phenomena like sarcasm or humor.

- **Cross-domain Testing**: If applicable, test your model across different domains or genres (e.g., comparing performance on news articles versus social media posts).
