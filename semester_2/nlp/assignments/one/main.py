import logging
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams
from collections import defaultdict, Counter
from random import choices
import numpy as np
import nltk
import re

nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams
from collections import defaultdict, Counter
from random import choices
import numpy as np
import re

def log_function_entry_exit(func):
    def wrapper(*args, **kwargs):
        print(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            print(f"Exiting {func.__name__} with result {result}")
            return result
        except Exception as e:
            raise Exception(f"Exception in {func.__name__}: {e}")
    return wrapper

@log_function_entry_exit
def preprocess_corpus(corpus):
    try:
        cleaned_text = re.sub(r'[^\w\s]', '', corpus)
        tokens = word_tokenize(cleaned_text.lower())
        print(f"Preprocessed {len(tokens)} tokens.")
        return tokens
    except Exception as e:
        raise Exception(f"Error in preprocess_corpus: {e}")

@log_function_entry_exit
def build_bigram_model(tokens, smoothing_factor=1):
    print("Starting build_bigram_model")
    try:
        bigram_counts = Counter(zip(tokens, tokens[1:]))
        vocab = set(tokens)
        vocab_size = len(vocab)

        total_counts = Counter(tokens)

        bigram_probs = {}
        for (w1, w2), count in bigram_counts.items():
            if w1 not in bigram_probs:
                bigram_probs[w1] = {}
            bigram_probs[w1][w2] = (count + smoothing_factor) / (total_counts[w1] + smoothing_factor * vocab_size)

        return bigram_probs, vocab_size
    except Exception as e:
        print(f"Error in build_bigram_model: {e}")
        raise
    finally:
        print("Exiting build_bigram_model")

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
        print(f"Error in generate_sentence: {e}")
        raise

@log_function_entry_exit
def evaluate_test_set(bigram_probs, test_set, vocab_size):
    try:
        sentence_probs = []
        for sentence in test_set:
            if len(sentence) < 2:
                print("Skipping sentence with less than 2 words")
                continue

            sentence_prob = 1.0
            for i in range(1, len(sentence)):
                second_word = sentence[i]
                first_word = sentence[i - 1]
                print(f"Evaluating bigram: [{first_word} -> {second_word}]")

                cond_prob = bigram_probs[first_word].get(second_word, 1 / vocab_size)
                sentence_prob *= cond_prob
                print(f"Conditional probability: {cond_prob}")

            sentence_probs.append(sentence_prob)
            print(f"Sentence probability: {sentence_prob}")

        avg_prob = np.mean(sentence_probs)
        std_dev = np.std(sentence_probs)

        return avg_prob, std_dev
    except Exception as e:
        raise Exception(f"Error in evaluate_test_set: {e}")

@log_function_entry_exit
def build_trigram_model(tokens, smoothing_factor=1):
    try:
        trigram_counts = Counter(zip(tokens, tokens[1:], tokens[2:]))
        vocab = set(tokens)
        vocab_size = len(vocab)

        bigram_counts = Counter(zip(tokens, tokens[1:]))

        trigram_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for (w1, w2, w3), count in trigram_counts.items():
            trigram_probs[w1][w2][w3] = (count + smoothing_factor) / (bigram_counts[(w1, w2)] + smoothing_factor * vocab_size)

        return trigram_probs, vocab_size
    except Exception as e:
        raise Exception(f"Error in build_trigram_model: {e}")

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
        raise Exception(f"Error in generate_sentence_trigram: {e}")

@log_function_entry_exit
def evaluate_test_set_trigram(trigram_probs, test_set, vocab_size):
    try:
        sentence_probs = []
        for sentence in test_set:
            if len(sentence) < 3:
                print("Skipping sentence with less than 3 words")
                continue

            sentence_prob = 1.0
            for i in range(2, len(sentence)):
                third_word = sentence[i]
                second_word = sentence[i - 1]
                first_word = sentence[i - 2]
                print(f"Evaluating trigram: [{first_word} -> {second_word} -> {third_word}]")

                cond_prob = trigram_probs[first_word][second_word].get(third_word, 1 / vocab_size)
                sentence_prob *= cond_prob
                print(f"Conditional probability: {cond_prob}")

            sentence_probs.append(sentence_prob)
            print(f"Sentence probability: {sentence_prob}")

        avg_prob = np.mean(sentence_probs)
        std_dev = np.std(sentence_probs)

        return avg_prob, std_dev
    except Exception as e:
        raise Exception(f"Error in evaluate_test_set_trigram: {e}")

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
        raise Exception(f"Error in calculate_perplexity: {e}")

try:
    with open("eng_news_2019_10K-sentences.txt", "r") as f:
        corpus = f.read()
    print("Corpus loaded successfully.")
except Exception as e:
    raise Exception(f"Error reading corpus file: {e}")

tokens = preprocess_corpus(corpus)

split_index = int(0.5 * len(tokens))
train_tokens = tokens[:split_index]
test_tokens = tokens[split_index:]
print(f"Training set size: {len(train_tokens)}")
print(f"Test set size: {len(test_tokens)}")

bigram_probs, vocab_size = build_bigram_model(train_tokens)
trigram_probs, vocab_size_trigram = build_trigram_model(train_tokens)

for i in range(10):
    try:
        generated_sentence = generate_sentence(bigram_probs, "the", vocab_size)
        print(f"Generated sentence {i+1} (Bigram): {' '.join(generated_sentence)}\n\n")
    except Exception as e:
        raise Exception(f"Error generating sentence: {e}")

for i in range(10):
    try:
        generated_sentence_trigram = generate_sentence_trigram(trigram_probs, ("the", "economy"), vocab_size_trigram)
        print(f"Generated sentence {i+1} (Trigram): {' '.join(generated_sentence_trigram)}\n\n")
    except Exception as e:
        raise Exception(f"Error generating sentence: {e}")

test_set = [["the", "weather", "is", "sunny"], ["the", "economy", "is", "booming"]]
try:
    avg_prob, std_dev = evaluate_test_set(bigram_probs, test_set, vocab_size)
    print(f"Average Probability (Provided Test Set - Bigram): {avg_prob}")
    print(f"Standard Deviation (Provided Test Set - Bigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating provided test set: {e}")

curated_test_set = [["artificial", "intelligence", "revolution"]]
try:
    avg_prob, std_dev = evaluate_test_set(bigram_probs, curated_test_set, vocab_size)
    print(f"Average Probability (Curated Test Set - Bigram): {avg_prob}")
    print(f"Standard Deviation (Curated Test Set - Bigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating curated test set: {e}")

try:
    avg_prob, std_dev = evaluate_test_set_trigram(trigram_probs, test_set, vocab_size_trigram)
    print(f"Average Probability (Provided Test Set - Trigram): {avg_prob}")
    print(f"Standard Deviation (Provided Test Set - Trigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating provided test set: {e}")

try:
    avg_prob, std_dev = evaluate_test_set_trigram(trigram_probs, curated_test_set, vocab_size_trigram)
    print(f"Average Probability (Curated Test Set - Trigram): {avg_prob}")
    print(f"Standard Deviation (Curated Test Set - Trigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating curated test set: {e}")
