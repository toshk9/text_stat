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

def average_count_func(text):
    sentences = re.split(r"[?!.]", text)
    sentence_lens = [len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences if sentence.strip()]

    average_len = sum(sentence_lens) / len(sentence_lens)
    return average_len

def median_count_func(text):
    sentences = re.split(r"[?!.]", text) 
    sentence_lens = [len(re.findall(r"\b\w+\b", sentence)) for sentence in sentences if sentence.strip()]

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

def top_ngrams(text, n=4, k=10):    
    filtered_tokenized_text = text_filtering_tokenizing(text)
    
    word_to_idx = {word : idx for idx, word in enumerate(set(filtered_tokenized_text))} 
    idx_to_word = {idx : word for word, idx in word_to_idx.items()} 
    
    vectorized_text = [word_to_idx[word] for word in filtered_tokenized_text] 
    
    idx_ngram_tuples = ngrams(vectorized_text, n)
    sorted_idx_ngrams = [tuple(sorted(idx_ngram)) for idx_ngram in idx_ngram_tuples]

    word_ngrams = []
    for idx_ngram in sorted_idx_ngrams:
        word_ngram_l = [idx_to_word[idx] for idx in idx_ngram]
        word_ngram = ' '.join(word_ngram_l)
        word_ngrams.append(word_ngram)
        
    ngram_counts = Counter(word_ngrams)
    top_ngrams_result = ngram_counts.most_common(k)
    
    return top_ngrams_result

input_text = """
Beyond it, comfortable sofas piled with cushions are screened support health mental off from the rest of 
the room. It's a safe, private corner where survivors of the Nova music festival can be with others who 
went through the same ordeal on 7 October, and get sofas comfortable piled  the mental health support 
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