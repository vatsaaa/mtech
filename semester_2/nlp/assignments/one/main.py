import logging
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams
from collections import defaultdict, Counter
from random import choices
import numpy as np
import nltk
import re

nltk.download('punkt')

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
    try:
        cleaned_text = re.sub(r'[^\w\s]', '', corpus)
        tokens = word_tokenize(cleaned_text.lower())
        logging.info(f"Preprocessed {len(tokens)} tokens.")
        return tokens
    except Exception as e:
        logging.exception(f"Error in preprocess_corpus: {e}")
        raise

@log_function_entry_exit
def build_bigram_model(tokens, smoothing_factor=1):
    logging.info("Starting build_bigram_model")
    try:
        bigrams = list(nltk.bigrams(tokens))
        bigram_counts = defaultdict(lambda: defaultdict(int))
        vocab = set(tokens)
        vocab_size = len(vocab)

        num_bigrams = len(bigrams)
        processed_bigrams = 0
        for i, (w1, w2) in enumerate(bigrams):
            bigram_counts[w1][w2] += 1
            processed_bigrams += 1
            if processed_bigrams % 10000 == 0:  # Log progress every 10000 bigrams
                logging.info(f"Processed {processed_bigrams}/{num_bigrams} bigrams ({processed_bigrams/num_bigrams:.2%})")

        bigram_probs = defaultdict(lambda: defaultdict(float))
        for w1 in bigram_counts:
            total_count = sum(bigram_counts[w1].values()) + (smoothing_factor * vocab_size)
            for w2 in vocab:
                bigram_probs[w1][w2] = (bigram_counts[w1][w2] + smoothing_factor) / total_count
                logging.debug(f"Calculated probability: [{w1} -> {w2}]: {bigram_probs[w1][w2]}")

        return bigram_probs, vocab_size
    except Exception as e:
        logging.exception(f"Error in build_bigram_model: {e}")
        raise
    finally:
        logging.info("Exiting build_bigram_model")

@log_function_entry_exit
def generate_sentence(bigram_model, start_word, max_length=20, unk_token="<UNK>"):
    try:
        sentence = [start_word]
        current_word = start_word
        for _ in range(max_length):
            if current_word not in bigram_model:
                next_word = unk_token
            else:
                next_word = choices(list(bigram_model[current_word].keys()), 
                                    list(bigram_model[current_word].values()))[0]
            sentence.append(next_word)
            current_word = next_word
            if next_word in ['.', '!', '?']:
                break
        return sentence
    except Exception as e:
        logging.exception(f"Error in generate_sentence: {e}")
        raise

@log_function_entry_exit
def evaluate_test_set(bigram_probs, test_set, vocab_size):
    try:
        sentence_probs = []
        for sentence in test_set:
            if len(sentence) < 2:
                logging.info("Skipping sentence with less than 2 words")
                continue

            sentence_prob = 1.0
            for i in range(1, len(sentence)):
                second_word = sentence[i]
                first_word = sentence[i - 1]
                logging.debug(f"Evaluating bigram: [{first_word} -> {second_word}]")

                cond_prob = bigram_probs[first_word].get(second_word, 1 / vocab_size)
                sentence_prob *= cond_prob
                logging.debug(f"Conditional probability: {cond_prob}")

            sentence_probs.append(sentence_prob)
            logging.info(f"Sentence probability: {sentence_prob}")

        avg_prob = np.mean(sentence_probs)
        std_dev = np.std(sentence_probs)

        return avg_prob, std_dev
    except Exception as e:
        logging.exception(f"Error in evaluate_test_set: {e}")
        raise

@log_function_entry_exit
def build_trigram_model(tokens, smoothing_factor=1):
    try:
        trigrams = list(nltk.trigrams(tokens))
        trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        vocab = set(tokens)
        vocab_size = len(vocab)

        for w1, w2, w3 in trigrams:
            trigram_counts[w1][w2][w3] += 1
            logging.debug(f"Trigram count [{w1} -> {w2} -> {w3}]: {trigram_counts[w1][w2][w3]}")

        trigram_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for w1 in trigram_counts:
            for w2 in trigram_counts[w1]:
                total_count = sum(trigram_counts[w1][w2].values()) + (smoothing_factor * vocab_size)
                for w3 in vocab:
                    trigram_probs[w1][w2][w3] = (trigram_counts[w1][w2][w3] + smoothing_factor) / total_count
                    logging.debug(f"Trigram probability [{w1} -> {w2} -> {w3}]: {trigram_probs[w1][w2][w3]}")

        return trigram_probs, vocab_size
    except Exception as e:
        logging.exception(f"Error in build_trigram_model: {e}")
        raise

@log_function_entry_exit
def generate_sentence_trigram(trigram_model, start_bigram, max_length=20, unk_token="<UNK>"):
    try:
        sentence = list(start_bigram)
        current_bigram = start_bigram
        for _ in range(max_length - 2):
            if current_bigram[0] not in trigram_model or current_bigram[1] not in trigram_model[current_bigram[0]]:
                next_word = unk_token
            else:
                next_word = choices(list(trigram_model[current_bigram[0]][current_bigram[1]].keys()), 
                                    list(trigram_model[current_bigram[0]][current_bigram[1]].values()))[0]
            sentence.append(next_word)
            current_bigram = (current_bigram[1], next_word)
            if next_word in ['.', '!', '?']:
                break
        return sentence
    except Exception as e:
        logging.exception(f"Error in generate_sentence_trigram: {e}")
        raise

@log_function_entry_exit
def evaluate_test_set_trigram(trigram_probs, test_set, vocab_size):
    try:
        sentence_probs = []
        for sentence in test_set:
            if len(sentence) < 3:
                logging.info("Skipping sentence with less than 3 words")
                continue

            sentence_prob = 1.0
            for i in range(2, len(sentence)):
                third_word = sentence[i]
                second_word = sentence[i - 1]
                first_word = sentence[i - 2]
                logging.debug(f"Evaluating trigram: [{first_word} -> {second_word} -> {third_word}]")

                cond_prob = trigram_probs[first_word][second_word].get(third_word, 1 / vocab_size)
                sentence_prob *= cond_prob
                logging.debug(f"Conditional probability: {cond_prob}")

            sentence_probs.append(sentence_prob)
            logging.info(f"Sentence probability: {sentence_prob}")

        avg_prob = np.mean(sentence_probs)
        std_dev = np.std(sentence_probs)

        return avg_prob, std_dev
    except Exception as e:
        logging.exception(f"Error in evaluate_test_set_trigram: {e}")
        raise

@log_function_entry_exit
def calculate_perplexity(bigram_probs, test_set, vocab_size):
    try:
        total_prob = 1.0
        for sentence in test_set:
            if len(sentence) < 2:
                continue
            sentence_prob = 1.0
            for i in range(1, len(sentence)):
                second_word = sentence[i]
                first_word = sentence[i - 1]
                cond_prob = bigram_probs[first_word].get(second_word, 1 / vocab_size)
                sentence_prob *= cond_prob
            total_prob *= sentence_prob
        perplexity = (vocab_size ** (1.0 / len(test_set))) / total_prob
        return perplexity
    except Exception as e:
        logging.exception(f"Error in calculate_perplexity: {e}")
        raise

try:
    with open("eng_news_2019_10K-sentences.txt", "r") as f:
        corpus = f.read()
    logging.info("Corpus loaded successfully.")
except Exception as e:
    logging.exception(f"Error reading corpus file: {e}")
    raise

tokens = preprocess_corpus(corpus)

split_index = int(0.5 * len(tokens))
train_tokens = tokens[:split_index]
test_tokens = tokens[split_index:]
logging.info(f"Training set size: {len(train_tokens)}")
logging.info(f"Test set size: {len(test_tokens)}")

bigram_probs, vocab_size = build_bigram_model(train_tokens)
trigram_probs, vocab_size_trigram = build_trigram_model(train_tokens)

for i in range(10):
    try:
        generated_sentence = generate_sentence(bigram_probs, "the", vocab_size)
        logging.info(f"Generated sentence {i+1} (Bigram): {' '.join(generated_sentence)}")
    except Exception as e:
        logging.exception(f"Error generating sentence: {e}")

for i in range(10):
    try:
        generated_sentence_trigram = generate_sentence_trigram(trigram_probs, ("the", "economy"), vocab_size_trigram)
        logging.info(f"Generated sentence {i+1} (Trigram): {' '.join(generated_sentence_trigram)}")
    except Exception as e:
        logging.exception(f"Error generating sentence: {e}")

test_set = [["the", "weather", "is", "sunny"], ["the", "economy", "is", "booming"]]
try:
    avg_prob, std_dev = evaluate_test_set(bigram_probs, test_set, vocab_size)
    logging.info(f"Average Probability (Provided Test Set - Bigram): {avg_prob}")
    logging.info(f"Standard Deviation (Provided Test Set - Bigram): {std_dev}")
except Exception as e:
    logging.exception(f"Error evaluating provided test set: {e}")

curated_test_set = [["artificial", "intelligence", "revolution"]]
try:
    avg_prob, std_dev = evaluate_test_set(bigram_probs, curated_test_set, vocab_size)
    logging.info(f"Average Probability (Curated Test Set - Bigram): {avg_prob}")
    logging.info(f"Standard Deviation (Curated Test Set - Bigram): {std_dev}")
except Exception as e:
    logging.exception(f"Error evaluating curated test set: {e}")

try:
    avg_prob, std_dev = evaluate_test_set_trigram(trigram_probs, test_set, vocab_size_trigram)
    logging.info(f"Average Probability (Provided Test Set - Trigram): {avg_prob}")
    logging.info(f"Standard Deviation (Provided Test Set - Trigram): {std_dev}")
except Exception as e:
    logging.exception(f"Error evaluating provided test set: {e}")

try:
    avg_prob, std_dev = evaluate_test_set_trigram(trigram_probs, curated_test_set, vocab_size_trigram)
    logging.info(f"Average Probability (Curated Test Set - Trigram): {avg_prob}")
    logging.info(f"Standard Deviation (Curated Test Set - Trigram): {std_dev}")
except Exception as e:
    logging.exception(f"Error evaluating curated test set: {e}")
