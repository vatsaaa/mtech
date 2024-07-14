from collections import defaultdict, Counter
import logging, nltk, re
from nltk.tokenize import word_tokenize
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
    """
    Preprocess the corpus by cleaning and tokenizing.

    Args:
        corpus (str): The input text corpus.

    Returns:
        list: List of tokens.
    """
    try:
        cleaned_text = re.sub(r'[^\w\s]', '', corpus)
        print(f"Cleaned text: {type(cleaned_text)}")
        print(f"Cleaned text: {cleaned_text}")
        tokens = word_tokenize(cleaned_text.lower(), treebank=True)
        print(f"Preprocessed {len(tokens)} tokens.")
        return tokens
    except Exception as e:
        raise Exception(f"Error in preprocess_corpus: {e}")

def remove_punctuation(text):
    """
    Remove punctuation from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: Text without punctuation.
    """
    punctuation = ".!\"#$%&()*+,/:;<=>?@[]\\^_{|}~—…´«»ʹ͵ͺ͵΄˂˃‘’“”„‟☀☁☂☃☄⚡️✨❣❤⚡️✨❣❤"  # Define punctuation characters
    no_punct = "".join([char for char in text if char not in punctuation])
    return no_punct

def preprocess_corpus_for_ngram(corpus, ngram, start_tag = '<s> ', end_tag = ' </s>'):
    """
    Preprocess the corpus for n-gram model by cleaning, tokenizing, and adding start/end tags.

    Args:
        corpus (str): The input text corpus.
        ngram (int): The n-gram level (e.g., 2 for bigram, 3 for trigram).
        start_tag (str, optional): The start tag to add. Defaults to '<s> '.
        end_tag (str, optional): The end tag to add. Defaults to ' </s>'.

    Returns:
        list: List of tokens with added start and end tags.
    """
    try:

        start_tag = start_tag * (ngram -1)

        cleaned_text = re.sub(r'\d+\t', '', corpus)
        cleaned_text = remove_punctuation(cleaned_text)
        pattern = r"(^|\n)([^\n]+)"
        cleaned_text = re.sub(r'(^|\n)([^\n]+)', rf"\1{start_tag}\2{end_tag}", cleaned_text)
        print("\n".join(cleaned_text.split('\n')[:10]))
        tokens = cleaned_text.lower().split()
        print(f"Preprocessed {len(tokens)} tokens.")
        return tokens
    except Exception as e:
        raise Exception(f"Error in preprocess_corpus: {e}")

@log_function_entry_exit
def build_unigram_model(tokens, smoothing_factor=1):
    """
    Build a unigram model with Laplace smoothing.

    Args:
        tokens (list): List of tokens from the corpus.
        smoothing_factor (int, optional): Smoothing factor for Laplace smoothing. Defaults to 1.

    Returns:
        tuple: Dictionary of unigram probabilities and vocabulary size.
    """
    try:
        unigram_counts = Counter(tokens)
        vocab = set(tokens)
        vocab_size = len(vocab)

        total_counts = sum(unigram_counts.values())

        unigram_probs = {}
        for w1, count in unigram_counts.items():
            #if w1 not in bigram_probs:
            #    unigram_probs[w1] = {}
            unigram_probs[w1] = (count + smoothing_factor) / (total_counts + smoothing_factor * vocab_size)

        return unigram_probs, vocab_size
    except Exception as e:
        print(f"Error in build_unigram_model: {e}")
        raise
    finally:
        print("Exiting build_unigram_model")

@log_function_entry_exit
def build_bigram_model(tokens, smoothing_factor=1):
    """
    Build a bigram model with Laplace smoothing.

    Args:
        tokens (list): List of tokens from the corpus.
        smoothing_factor (int, optional): Smoothing factor for Laplace smoothing. Defaults to 1.

    Returns:
        tuple: Dictionary of bigram probabilities and vocabulary size.
    """
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

#Generates a sentence using a unigram model with Laplace smoothing.
def generate_unigram_sentence(word_probabilities, start_word, max_length=20):
    """
    Generate a sentence using a unigram model with Laplace smoothing.

    Args:
        word_probabilities (dict): Dictionary of word probabilities.
        start_word (str): Starting word for the sentence.
        max_length (int, optional): Maximum length of the sentence. Defaults to 20.

    Returns:
        list: Generated sentence as a list of words.
    """
    initials = '<s>'
    sentence = [initials]
    sentence.append(start_word)
    current_word = start_word
    for _ in range(max_length):
        if current_word not in word_probabilities:
            next_word = '<UNK>'
        else:
            next_word = choices(list(word_probabilities.keys()), weights=list(word_probabilities.values()))[0]
            sentence.append(next_word)
            current_word = next_word
            if next_word in ['</s>']:
                break
    return sentence

def generate_sentence(bigram_model, start_word, max_length=20, unk_token="<UNK>"):
    """
    Generate a sentence using a bigram model.

    Args:
        bigram_model (dict): Dictionary of bigram probabilities.
        start_word (str): Starting word for the sentence.
        max_length (int, optional): Maximum length of the sentence. Defaults to 20.
        unk_token (str, optional): Token to use for unknown words. Defaults to "<UNK>".

    Returns:
        list: Generated sentence as a list of words.
    """
    try:
        initials = '<s>'
        sentence = [initials]
        sentence.append(start_word)
        current_word = start_word

        for _ in range(max_length):
            if current_word not in bigram_model:
                next_word = unk_token
            else:
                #keys_of_current_word = list(bigram_model[current_word].keys())
                #max_value = max(bigram_model[current_word].values())
                next_word = max(bigram_model[current_word], key=bigram_model[current_word].get)

                #next_word = choices(list(bigram_model[current_word].keys()),
                #                   list(bigram_model[current_word].values()))[0]
            sentence.append(next_word)
            current_word = next_word
            if next_word in ['</s>']:
                break
        return sentence
    except Exception as e:
        print(f"Error in generate_sentence: {e}")
        raise

@log_function_entry_exit
def evaluate_test_set(bigram_probs, test_set, vocab_size):
    """
    Evaluate the test set using the bigram model.

    Args:
        bigram_probs (dict): Dictionary of bigram probabilities.
        test_set (list): List of test sentences (each sentence is a list of words).
        vocab_size (int): Vocabulary size.

    Returns:
        tuple: Average probability and standard deviation of the test set.
    """
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

        avg_prob = numpy.mean(sentence_probs)
        std_dev = numpy.std(sentence_probs)

        return avg_prob, std_dev
    except Exception as e:
        raise Exception(f"Error in evaluate_test_set: {e}")
    
@log_function_entry_exit
def build_trigram_model(tokens, smoothing_factor=1):
    """
    Build a trigram model with Laplace smoothing.

    Args:
        tokens (list): List of tokens from the corpus.
        smoothing_factor (int, optional): Smoothing factor for Laplace smoothing. Defaults to 1.

    Returns:
        dict: Dictionary of trigram probabilities.
    """
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

def generate_trigram_sentence(trigram_model, start_bigram, max_length=20, unk_token="<UNK>"):
    try:
        initials = '<s>'
        sentence = [initials]
        sentence.extend(list(start_bigram))

        current_bigram = start_bigram
        for _ in range(max_length - 2):
            if current_bigram[0] not in trigram_model or current_bigram[1] not in trigram_model[current_bigram[0]]:
                next_word = unk_token
            else:

                next_word = max(trigram_model[current_bigram[0]][current_bigram[1]], key=trigram_model[current_bigram[0]][current_bigram[1]].get)
                #next_word = choices(list(trigram_model[current_bigram[0]][current_bigram[1]].keys()),
                #                    list(trigram_model[current_bigram[0]][current_bigram[1]].values()))[0]
            sentence.append(next_word)
            current_bigram = (current_bigram[1], next_word)
            if next_word in ['</s>']:
                sentence.append('</s>')
                break
        return sentence
    except Exception as e:
        raise Exception(f"Error in generate_trigram_sentence: {e}")

@log_function_entry_exit
def evaluate_test_set_trigram(trigram_probs, test_set, vocab_size):
    """
    Evaluate the test set using the trigram model.

    Args:
        trigram_probs (dict): Dictionary of trigram probabilities.
        test_set (list): List of test sentences (each sentence is a list of words).
        vocab_size (int): Vocabulary size.

    Returns:
        tuple: Average probability and standard deviation of the test set.
    """
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

        avg_prob = numpy.mean(sentence_probs)
        std_dev = numpy.std(sentence_probs)

        return avg_prob, std_dev
    except Exception as e:
        raise Exception(f"Error in evaluate_test_set_trigram: {e}")

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

#Training tokens for bigram
tokens = preprocess_corpus_for_ngram(corpus,2)

split_index = int(0.8 * len(tokens))
train_tokens = tokens[:split_index]
print(f"Training set size: {train_tokens}")
test_tokens = tokens[split_index:]
print(f"Training set size: {len(train_tokens)}")
print(f"Test set size: {len(test_tokens)}")

#Training tokens for trigram
trigram_tokens = preprocess_corpus_for_ngram(corpus,3)

trigram_split_index = int(0.8 * len(trigram_tokens))
train_trigram_tokens = trigram_tokens[:trigram_split_index]
print(f"Training set size: {train_trigram_tokens}")
test_trigram_tokens = trigram_tokens[trigram_split_index:]
print(f"Training set size: {len(train_trigram_tokens)}")
print(f"Test set size: {len(test_trigram_tokens)}")

unigram_probs, vocab_size = build_unigram_model(train_tokens)

bigram_probs, vocab_size = build_bigram_model(train_tokens)

trigram_probs, vocab_size_trigram = build_trigram_model(trigram_tokens)
print(trigram_probs)

def pick_random_token(tokens, exclude_tokens):

  filtered_tokens = []
  # Create a filtered list excluding the specified tokens
  for i, token in enumerate(tokens):
    if token not in exclude_tokens and (tokens[i-1] is None or tokens[i-1] == "<s>"):
      filtered_tokens.append(token)

  #filtered_tokens = [token for i, token in tokens if token not in exclude_tokens)]

  # Check if there are any tokens left after exclusion
  if not filtered_tokens:
    raise ValueError("No tokens left after exclusion!")

  # Choose a random token from the filtered list
  random_token = choice(filtered_tokens)
  return random_token

#Generate sentence using unigram model
for i in range(10):
    try:
        exclude_tokens = ["<s>", "</s>"]
        random_token = pick_random_token(train_tokens, exclude_tokens)
        generated_sentence = generate_unigram_sentence(unigram_probs, random_token)
        print(f"Generated sentence {i+1} (Unigram): {' '.join(generated_sentence)}")
    except Exception as e:
        raise Exception(f"Error generating sentence: {e}")
    
for i in range(10):
    try:
        exclude_tokens = ["<s>", "</s>"]
        random_token = pick_random_token(train_tokens, exclude_tokens)
        generated_sentence = generate_sentence(bigram_probs, random_token, vocab_size)
        print(f"Generated sentence {i+1} (Bigram): {' '.join(generated_sentence)}")
    except Exception as e:
        raise Exception(f"Error generating sentence: {e}")
    
# generated_sentence_trigram = generate_sentence_trigram(trigram_probs, ("<s>", "the"), vocab_size_trigram)
# print(f"Generated sentence 0 (Trigram): {' '.join(generated_sentence_trigram)}")
for i in range(10):
    try:
        exclude_tokens = ["<s>", "</s>"]
        random_token = pick_random_token(trigram_tokens, exclude_tokens)
        start_bigram = ("<s>", random_token)
        generated_sentence = generate_trigram_sentence(trigram_probs, start_bigram, vocab_size_trigram)
        print(f"Generated sentence {i+1} (Trigram): {' '.join(generated_sentence)}")
    except Exception as e:
        raise Exception(f"Error generating sentence: {e}")

#Evaluate Test set for Bigram
test_set = [["the", "weather", "is", "sunny"], ["the", "economy", "is", "booming"]]
try:
    avg_prob, std_dev = evaluate_test_set(bigram_probs, test_set, vocab_size)
    print(f"Average Probability (Provided Test Set - Bigram): {avg_prob}")
    print(f"Standard Deviation (Provided Test Set - Bigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating provided test set: {e}")

#Evaluate Curated set for Bigram
curated_test_set = [["artificial", "intelligence", "revolution"]]
try:
    avg_prob, std_dev = evaluate_test_set(bigram_probs, curated_test_set, vocab_size)
    print(f"Average Probability (Curated Test Set - Bigram): {avg_prob}")
    print(f"Standard Deviation (Curated Test Set - Bigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating curated test set: {e}")

#Evaluate Test set for Trigram
try:
    avg_prob, std_dev = evaluate_test_set_trigram(trigram_probs, test_set, vocab_size_trigram)
    print(f"Average Probability (Provided Test Set - Trigram): {avg_prob}")
    print(f"Standard Deviation (Provided Test Set - Trigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating provided test set: {e}")

#Evaluate Test set for Trigram
try:
    avg_prob, std_dev = evaluate_test_set_trigram(trigram_probs, curated_test_set, vocab_size_trigram)
    print(f"Average Probability (Curated Test Set - Trigram): {avg_prob}")
    print(f"Standard Deviation (Curated Test Set - Trigram): {std_dev}")
except Exception as e:
    raise Exception(f"Error evaluating curated test set: {e}")
