import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import ConditionalFreqDist, FreqDist
import numpy as np
from re import sub
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_function_entry_exit(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"Exiting {func.__name__} with result {result}")
            return result
        except Exception as e:
            logging.exception(f"Exception in {func.__name__}: {e}")
            raise
    return wrapper

@log_function_entry_exit
def preprocess_corpus(corpus):
    """
    Preprocesses the corpus text by cleaning, tokenizing, and lowercasing.

    Args:
    corpus: String containing the raw corpus text.

    Returns:
    A list of preprocessed tokens (words).
    """
    try:
        cleaned_text = sub(r'[^\w\s]', '', corpus)
        tokens = word_tokenize(cleaned_text.lower())
        return tokens
    except Exception as e:
        logging.exception(f"Error in preprocess_corpus: {e}")
        raise

@log_function_entry_exit
def build_bigram_model(tokens, smoothing_factor=1):
    """
    Builds a bigram model from the preprocessed tokens with Laplace smoothing.

    Args:
    tokens: A list of preprocessed tokens (words).
    smoothing_factor: The value to add for smoothing (default: 1).

    Returns:
    A ConditionalFreqDist object representing the bigram probabilities with smoothing.
    """
    try:
        bigrams = list(nltk.bigrams(tokens))
        bigram_counts = defaultdict(lambda: defaultdict(int))

        # Track intermediate values for debugging
        print("Bigram Counts (Before Smoothing):")
        for w1, w2 in bigrams:
            bigram_counts[w1][w2] += 1
            print(f"{w1} -> {w2}: {bigram_counts[w1][w2]}")  # Print bigram counts

        smoothed_counts = {}
        vocab = set(tokens)
        for w1 in bigram_counts:
            smoothed_counts[w1] = FreqDist(bigram_counts[w1])
            for w2 in vocab:
                smoothed_counts[w1][w2] += smoothing_factor

        # Track intermediate values for debugging
        print("Smoothed Counts:")
        for w1, inner_dict in smoothed_counts.items():
            print(f"{w1}: {inner_dict.items()}")  # Print smoothed counts

        cfd = ConditionalFreqDist((w1, w2_count)
                                  for w1, w2_counts in smoothed_counts.items()
                                  for w2_count in w2_counts.items())

        return cfd
    except Exception as e:
        logging.exception(f"Error in build_bigram_model: {e}")
        raise

@log_function_entry_exit
def generate_sentence(bigram_model, start_word, max_length=20):
    """
    Generates a sentence using the bigram model.

    Args:
    bigram_model: A ConditionalFreqDist object representing the bigram probabilities.
    start_word: The starting word for the sentence.
    max_length: The maximum desired length of the sentence.

    Returns:
    A generated sentence as a list of words.
    """
    try:
        sentence = [start_word]
        current_word = start_word
        for _ in range(max_length):
            next_word_probs = bigram_model[current_word]
            next_word = next_word_probs.max()
            sentence.append(next_word)
            current_word = next_word
            if next_word in ['.', '!', '?']:
                break
        return sentence
    except Exception as e:
        logging.exception(f"Error in generate_sentence: {e}")
        raise

@log_function_entry_exit
def evaluate_test_set(bigram_model, test_set):
    """
    Evaluates the model on a test set by calculating average and standard deviation
    of sentence probabilities.

    Args:
    bigram_model: A ConditionalFreqDist object representing the bigram probabilities.
    test_set: A list of preprocessed test sentences.

    Returns:
    A tuple containing the average and standard deviation of sentence probabilities.
    """
    try:
        sentence_probs = []
        for sentence in test_set:
            sentence_prob = 1.0

            if len(sentence) < 2:
                logging.info("Skipping sentence with less than 2 words")
                continue

            for i in range(1, len(sentence)):
                second_word = sentence[i]
                first_word = sentence[i - 1]

                if first_word in bigram_model:
                    cond_prob = bigram_model[first_word].freq(second_word)
                else:
                    cond_prob = 1 / len(bigram_model)

                sentence_prob *= cond_prob
            sentence_probs.append(sentence_prob)
        avg_prob = sum(sentence_probs) / len(sentence_probs)
        std_dev = np.std(sentence_probs)

        return (avg_prob, std_dev)
    except Exception as e:
        logging.exception(f"Error in evaluate_test_set: {e}")
        raise


# Load English news corpus as a string
try:
    with open("eng_news_2019_10K-sentences.txt", "r") as f:
        corpus = f.read()
except Exception as e:
    logging.exception(f"Error reading corpus file: {e}")
    raise

# Preprocess the corpus
tokens = preprocess_corpus(corpus)

# Build the bigram model
bigram_model = build_bigram_model(tokens)

# Generate 10 sentences
for _ in range(10):
    try:
        generated_sentence = generate_sentence(bigram_model, "the")
        logging.info(f"Generated sentence: {' '.join(generated_sentence)}")
    except Exception as e:
        logging.exception(f"Error generating sentence: {e}")

# Evaluate the model on provided test set (replace with your test set)
test_set = [["the", "weather", "is", "sunny"], ["the", "economy", "is", "booming"]]
try:
    avg_prob, std_dev = evaluate_test_set(bigram_model, test_set)
    logging.info(f"Average Probability (Provided Test Set): {avg_prob}")
    logging.info(f"Standard Deviation (Provided Test Set): {std_dev}")
except Exception as e:
    logging.exception(f"Error evaluating provided test set: {e}")

# Create your own curated test set
curated_test_set = [["artificial", "intelligence", "revolution"]]
try:
    avg_prob, std_dev = evaluate_test_set(bigram_model, curated_test_set)
    logging.info(f"Average Probability (Curated Test Set): {avg_prob}")
    logging.info(f"Standard Deviation (Curated Test Set): {std_dev}")
except Exception as e:
    logging.exception(f"Error evaluating curated test set: {e}")
