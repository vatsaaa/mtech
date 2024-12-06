from collections import defaultdict, Counter
import logging
import nltk
import re
from random import choice, choices
import numpy

nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_function_entry_exit(func):
    """
    Decorator to log the entry and exit of a function.

    Args:
        func (function): The function to be wrapped.

    Returns:
        function: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        logging.info(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"Exiting {func.__name__} with result {result}")
            return result
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {e}")
            raise
    return wrapper

def remove_punctuation(text):
    """
    Remove punctuation from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: Text without punctuation.
    """
    punctuation = r"[\.\!\"#\$%&\(\)\*\+\,\/\:\;\<=\>\?@\[\\\]\^_\{\|\}~—…´«»ʹ͵ͺ͵΄˂˃‘’“”„‟☀☁☂☃☄⚡️✨❣❤⚡️✨❣❤]"
    no_punct = re.sub(punctuation, '', text)
    return no_punct

def preprocess_corpus_for_ngram(corpus, ngram, start_tag='<s>', end_tag='</s>'):
    """
    Preprocess the corpus for n-gram model by cleaning, tokenizing, and adding start/end tags.

    Args:
        corpus (str): The input text corpus.
        ngram (int): The n-gram level (e.g., 2 for bigram, 3 for trigram).
        start_tag (str, optional): The start tag to add. Defaults to '<s>'.
        end_tag (str, optional): The end tag to add. Defaults to '</s>'.

    Returns:
        list: List of tokens with added start and end tags.
    """
    try:
        cleaned_text = re.sub(r'\d+\t', '', corpus)
        cleaned_text = remove_punctuation(cleaned_text)
        pattern = r"(^|\n)([^\n]+)"
        start_tag_str = (start_tag + ' ') * (ngram - 1)
        cleaned_text = re.sub(pattern, rf"\1{start_tag_str}\2 {end_tag}", cleaned_text)
        tokens = cleaned_text.lower().split()
        logging.info(f"Preprocessed {len(tokens)} tokens for {ngram}-gram model.")
        return tokens
    except Exception as e:
        logging.error(f"Error in preprocess_corpus_for_ngram: {e}")
        raise Exception(f"Error in preprocess_corpus_for_ngram: {e}")

# @log_function_entry_exit
def build_ngram_model(tokens, n, smoothing_factor=1):
    try:
        if n == 1:
            ngram_counts = Counter(tokens)
            vocab = set(tokens)
            vocab_size = len(vocab)

            total_counts = sum(ngram_counts.values())
            ngram_probs = {}

            for w1, count in ngram_counts.items():
                ngram_probs[w1] = (count + smoothing_factor) / (total_counts + smoothing_factor * vocab_size)
        
        elif n == 2:
            ngram_counts = Counter(zip(tokens, tokens[1:]))
            vocab = set(tokens)
            vocab_size = len(vocab)

            total_counts = Counter(tokens)
            ngram_probs = {}

            for (w1, w2), count in ngram_counts.items():
                if w1 not in ngram_probs:
                    ngram_probs[w1] = {}
                ngram_probs[w1][w2] = (count + smoothing_factor) / (total_counts[w1] + smoothing_factor * vocab_size)
        
        elif n == 3:
            ngram_counts = Counter(zip(tokens, tokens[1:], tokens[2:]))
            vocab = set(tokens)
            vocab_size = len(vocab)

            bigram_counts = Counter(zip(tokens, tokens[1:]))
            ngram_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

            for (w1, w2, w3), count in ngram_counts.items():
                ngram_probs[w1][w2][w3] = (count + smoothing_factor) / (bigram_counts[(w1, w2)] + smoothing_factor * vocab_size)

        return ngram_probs, vocab_size
    except Exception as e:
        raise Exception(f"Error in build_ngram_model: {e}")

def generate_ngram_sentence(ngram_model, start_words, max_length=20, unk_token="<UNK>"):
    """
    Generate a sentence using an n-gram model.

    Args:
        ngram_model (dict or defaultdict): Dictionary of n-gram probabilities (can be unigram, bigram, or trigram).
        start_words (tuple): Starting words for the sentence (tuple of n-1 words for bigram, n-2 for trigram).
        max_length (int, optional): Maximum length of the sentence. Defaults to 20.
        unk_token (str, optional): Token to use for unknown words. Defaults to "<UNK>".

    Returns:
        list: Generated sentence as a list of words.
    """
    try:
        sentence = list(start_words)
        current_words = start_words

        for _ in range(max_length - len(start_words)):
            if isinstance(ngram_model, dict) and all(isinstance(v, (int, float)) for v in ngram_model.values()):  # Unigram
                next_word = choices(list(ngram_model.keys()), weights=list(ngram_model.values()))[0]
                sentence.append(next_word)
                current_words = (next_word,)
            elif isinstance(ngram_model, dict) and all(isinstance(v, dict) for v in ngram_model.values()):  # Bigram
                current_bigram = current_words[-1]
                if current_bigram not in ngram_model:
                    next_word = unk_token
                else:
                    bigram_values = [float(sum(v.values())) if isinstance(v, defaultdict) else float(v) for v in ngram_model[current_bigram].values()]
                    next_word = choices(list(ngram_model[current_bigram].keys()), weights=bigram_values)[0]
                sentence.append(next_word)
                current_words = (current_bigram, next_word)
            elif (isinstance(ngram_model, dict) and
                  all(isinstance(v, dict) for v in ngram_model.values()) and
                  all(isinstance(vv, dict) for v in ngram_model.values() for vv in v.values())):  # Trigram
                current_trigram = current_words[-2:]
                if (current_trigram[0] not in ngram_model or
                        current_trigram[1] not in ngram_model[current_trigram[0]]):
                    next_word = unk_token
                else:
                    trigram_values = [float(sum(v.values())) if isinstance(v, defaultdict) else float(v) for v in ngram_model[current_trigram[0]][current_trigram[1]].values()]
                    next_word = choices(list(ngram_model[current_trigram[0]][current_trigram[1]].keys()), weights=trigram_values)[0]
                sentence.append(next_word)
                current_words = (current_trigram[1], next_word)

            if next_word == '</s>':
                break

        return sentence

    except Exception as e:
        logging.error(f"Error in generate_ngram_sentence: {e}")
        raise Exception(f"Error in generate_ngram_sentence: {e}")

@log_function_entry_exit
def evaluate_test_set(probs_dict, test_set, vocab_size, n):
    """
    Evaluate the test set using the n-gram model.

    Args:
        probs_dict (dict): Dictionary of n-gram probabilities.
        test_set (list): List of test sentences (each sentence is a list of words).
        vocab_size (int): Vocabulary size.
        n (int): Order of the n-gram model (1 for unigram, 2 for bigram, 3 for trigram).

    Returns:
        tuple: Average probability and standard deviation of the test set.
    """
    try:
        sentence_probs = []
        for sentence in test_set:
            if len(sentence) < n:
                print(f"Skipping sentence with less than {n} words")
                continue

            sentence_prob = 1.0
            for i in range(n - 1, len(sentence)):
                context = tuple(sentence[i - n + 1:i])  # Previous (n-1) words as context
                word = sentence[i]

                if context not in probs_dict or word not in probs_dict[context]:
                    # Handle unseen n-grams: use Laplace smoothing
                    print(f"Unseen n-gram: {context} -> {word}")
                    cond_prob = 1 / vocab_size
                else:
                    cond_prob = probs_dict[context][word]

                sentence_prob *= cond_prob

            sentence_probs.append(sentence_prob)
            print(f"Sentence probability: {sentence_prob}")

        avg_prob = numpy.mean(sentence_probs)
        std_dev = numpy.std(sentence_probs, ddof=1) if len(sentence_probs) > 1 else 0.0

        print(f"Number of sentences evaluated: {len(sentence_probs)}")
        print(f"Average Probability: {avg_prob}")
        print(f"Standard Deviation: {std_dev}")

        return float(avg_prob), float(std_dev)

    except Exception as e:
        raise Exception(f"Error in evaluate_test_set: {e}")

@log_function_entry_exit
def calculate_perplexity(bigram_probs, test_set, vocab_size):
    """
    Calculate the perplexity of a test set using the bigram model.

    Args:
        bigram_probs (dict): Dictionary of bigram probabilities.
        test_set (list): List of test sentences (each sentence is a list of words).
        vocab_size (int): Vocabulary size.

    Returns:
        float: Perplexity score of the test set.
    """
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

def pick_random_token(tokens, exclude_tokens):
    filtered_tokens = [token for token in tokens if token not in exclude_tokens]
    if not filtered_tokens:
        logging.error(f"Filtered out all tokens. Tokens: {tokens}, Exclude: {exclude_tokens}")
        raise ValueError("No tokens left after exclusion!")
    return choice(filtered_tokens)

#Training tokens for unigram
unigram_tokens = preprocess_corpus_for_ngram(corpus, 1)
split_index = int(0.8 * len(unigram_tokens))
train_unigram_tokens = unigram_tokens[:split_index]
# print(f"Training set size: {train_unigram_tokens}")
test_unigram_tokens = unigram_tokens[split_index:]
print(f"Unigram Training set size: {len(train_unigram_tokens)}")
print(f"Unigram Test set size: {len(test_unigram_tokens)}")

#Training tokens for bigram
bigram_tokens = preprocess_corpus_for_ngram(corpus, 2)
split_index = int(0.8 * len(bigram_tokens))
train_bigram_tokens = bigram_tokens[:split_index]
# print(f"Training set size: {train_bigram_tokens}")
test_bigram_tokens = bigram_tokens[split_index:]
print(f"Bigram Training set size: {len(train_bigram_tokens)}")
print(f"Bigram Test set size: {len(test_bigram_tokens)}")

trigram_tokens = preprocess_corpus_for_ngram(corpus, 3)
trigram_split_index = int(0.8 * len(trigram_tokens))
train_trigram_tokens = trigram_tokens[:trigram_split_index]
# print(f"Training set size: {train_trigram_tokens}")
test_trigram_tokens = trigram_tokens[trigram_split_index:]
print(f"Trigram Training set size: {len(train_trigram_tokens)}")
print(f"Trigram Test set size: {len(test_trigram_tokens)}")

unigram_probs, unigram_vocab_size = build_ngram_model(unigram_tokens, 1)
unigram_probs = dict(unigram_probs)

bigram_probs, bigram_vocab_size = build_ngram_model(bigram_tokens, 2)
bigram_probs = {key: dict(value) for key, value in bigram_probs.items()}

trigram_probs, trigram_vocab_size = build_ngram_model(trigram_tokens, 3)
trigram_probs = {key: dict(value) for key, value in trigram_probs.items()}

#Generate sentence using unigram model
for i in range(10):
    try:
        exclude_tokens = ["<s>", "</s>"]
        random_token = pick_random_token(train_unigram_tokens, exclude_tokens)
        generated_sentence_unigram = generate_ngram_sentence(unigram_probs, (random_token,), max_length=20, unk_token="<UNK>")
        print(f"Generated sentence {i+1} (Unigram): {' '.join(generated_sentence_unigram)}")
    except Exception as e:
        raise Exception(f"Error generating sentence: {e}")
    
#Generate sentence using bigram model
for i in range(10):
    try:
        exclude_tokens = ["<s>", "</s>"]
        random_token = pick_random_token(train_bigram_tokens, exclude_tokens)
        generated_sentence_bigram = generate_ngram_sentence(bigram_probs, (random_token,), max_length=20, unk_token="<UNK>")
        print(f"Generated sentence {i+1} (Bigram): {' '.join(generated_sentence_bigram)}")
    except Exception as e:
        raise Exception(f"Error generating sentence: {e}")
    
#Generate sentence using trigram model
for i in range(10):
    try:
        exclude_tokens = ["<s>", "</s>"]
        random_token = pick_random_token(train_trigram_tokens, exclude_tokens)
        start_bigram = ("<s>", random_token)
        generated_sentence_trigram = generate_ngram_sentence(trigram_probs, start_bigram, max_length=20, unk_token="<UNK>")
        print(f"Generated sentence {i+1} (Trigram): {' '.join(generated_sentence_trigram)}")
    except Exception as e:
        raise Exception(f"Error generating sentence: {e}")

test_set = [["the", "weather", "is", "sunny"], ["the", "economy", "is", "booming"]]
curated_test_set = [["artificial", "intelligence", "revolution"]]

#Evaluate Test set for Unigram
print("Evaluating Unigram with")
print("\ttest set")
try:
    avg_prob, std_dev = evaluate_test_set(unigram_probs, test_set, unigram_vocab_size, 1)
    print(f"Average Probability (Provided Test Set - Unigram): {avg_prob}")
    print(f"Standard Deviation (Provided Test Set - Unigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating provided test set: {e}")

print("\tcurated set")
#Evaluate Curated set for Unigram
try:
    avg_prob, std_dev = evaluate_test_set(unigram_probs, curated_test_set, unigram_vocab_size, 1)
    print(f"Average Probability (Curated Test Set - Unigram): {avg_prob}")
    print(f"Standard Deviation (Curated Test Set - Unigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating curated test set: {e}")

#Evaluate Test set for Bigram
print("Evaluating Bigram with")
print("\ttest set")
try:
    avg_prob, std_dev = evaluate_test_set(bigram_probs, test_set, bigram_vocab_size, 2)
    print(f"Average Probability (Provided Test Set - Bigram): {avg_prob}")
    print(f"Standard Deviation (Provided Test Set - Bigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating provided test set: {e}")

#Evaluate Curated set for Bigram
print("\tcurated set")
try:
    avg_prob, std_dev = evaluate_test_set(bigram_probs, curated_test_set, bigram_vocab_size, 2)
    print(f"Average Probability (Curated Test Set - Bigram): {avg_prob}")
    print(f"Standard Deviation (Curated Test Set - Bigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating curated test set: {e}")

#Evaluate Test set for Trigram
print("Evaluating Trigram with")
print("\ttest set")
try:
    avg_prob, std_dev = evaluate_test_set(trigram_probs, test_set, trigram_vocab_size, 3)
    print(f"Average Probability (Provided Test Set - Trigram): {avg_prob}")
    print(f"Standard Deviation (Provided Test Set - Trigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating provided test set: {e}")

#Evaluate curated set for Trigram
print("\tcurated set")
try:
    avg_prob, std_dev = evaluate_test_set(trigram_probs, curated_test_set, trigram_vocab_size, 3)
    print(f"Average Probability (Curated Test Set - Trigram): {avg_prob}")
    print(f"Standard Deviation (Curated Test Set - Trigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating curated test set: {e}")
