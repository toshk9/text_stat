import re
from collections import Counter
import nltk

nltk.download("stopwords")
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk import word_tokenize

import string


def count_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return Counter(words)

def text_sentences_lens(text):
    sentences = re.split(r"[?!.]", text)
    sentence_lens = [len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences[:-1]]
    return sentence_lens

def average_count_func(text):
    sentence_lens = text_sentences_lens(text)

    average_len = sum(sentence_lens) / len(sentence_lens)
    return average_len

def median_count_func(text):
    sentence_lens = text_sentences_lens(text)

    sorted_sentence_lens = sorted(sentence_lens) 
    if len(sentence_lens) % 2 == 0:
        median_count = (sorted_sentence_lens[len(sorted_sentence_lens) // 2 - 1] + sorted_sentence_lens[len(sorted_sentence_lens) // 2]) / 2
    else:    
        median_count = sorted_sentence_lens[len(sorted_sentence_lens) // 2]
    return median_count

def text_filtering_tokenizing(text):
    for p in string.punctuation:
        text = text.replace(p, "")

    text = text.replace("\n", "")    
    
    tokenized_text = word_tokenize(text.lower())
    
    stop_words = stopwords.words("english")
    
    filtered_text = [word for word in tokenized_text if word not in stop_words]
    return filtered_text

def ngrams(iterable, n):
    start = 0
    stop = n
    ngrams = []
    while stop <= len(iterable):
        ngrams.append(tuple(iterable[start : stop]))
        start += 1
        stop += 1
    return iter(ngrams)

def top_ngrams(text, n=4, k=10): # takes raw text with parameters n (n-gram length) and k (top k most frequently used)
    filtered_tokenized_text = text_filtering_tokenizing(text) # returned filtered and tokenized text
    
    word_to_idx = {word : idx for idx, word in enumerate(set(filtered_tokenized_text))} # dictionary like {"word" : idx}

    idx_to_word = {idx : word for word, idx in word_to_idx.items()} # dictionary like {idx : "word"}
    
    vectorized_text = [word_to_idx[word] for word in filtered_tokenized_text]  # converting words to unique indices
    
    idx_ngram_tuples = ngrams(vectorized_text, n) # n-grams of the vectorized text 
    sorted_idx_ngrams = [tuple(sorted(idx_ngram)) for idx_ngram in idx_ngram_tuples] # sorting indices inside n-gram

    word_ngrams = [] # empty list for literal n-grams (as a single string)
    for idx_ngram in sorted_idx_ngrams: # iteration over a list of numeric n-grams
        word_ngram_l = [idx_to_word[idx] for idx in idx_ngram] # converting indices to words using a dictionary idx_to_word
        word_ngram = ' '.join(word_ngram_l) # n-grams as word lists are converted to individual strings
        word_ngrams.append(word_ngram) # collecting all n-grams into one list
        
    ngram_counts = Counter(word_ngrams) # counting n-grams
    top_ngrams_result = ngram_counts.most_common(k) # getting top k most frequently used n-grams
    
    return top_ngrams_result # returning a list of tuples like: [("n-gram1", times1), ("n-gram2", times2), ...]

input_text = """
Beyond it, comfortable sofas piled with cushions are screened support health mental off from the rest of 
the room. It's a safe, private corner where survivors of the Nova music festival can be with others who 
went through the. same ordeal on 7 October, and get sofas comfortable piled  the mental health support 
many of them desperately need.
"""

word_counts = count_words(input_text)
print(f"Words and their frequencies:\n{word_counts}\n")

avg_words_per_sentence = average_count_func(input_text)
print(f"Average number of words per sentence: \n{avg_words_per_sentence:.2f}\n")

median_words_per_sentence = median_count_func(input_text)
print(f"Median number of words per sentence: \n{median_words_per_sentence:.2f}\n")

k_value = int(input("Enter K value for top-K word N-grams: "))
n_value = int(input("Enter N value for literal N-grams: "))

top_ngrams_result = top_ngrams(input_text, n_value, k_value)
print(f"\nTop-{k_value} most frequently repeated letter {n_value}-grams:\n{top_ngrams_result}\n")