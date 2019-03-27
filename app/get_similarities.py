import requests
import re
from bs4 import BeautifulSoup
import numpy as np
from numpy.linalg import norm
from bert_serving.client import BertClient
import json
from itertools import product
import io
import pickle
from collections import defaultdict

stopwords =    {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", \
                "you", "your", "yours", "yourself", "yourselves", "he", "him", \
                "his", "himself", "she", "her", "hers", "herself", "it", "its", \
                "itself", "they", "them", "their", "theirs", "themselves", "what", \
                "which", "who", "whom", "this", "that", "these", "those", "am", "is", \
                "are", "was", "were", "be", "been", "being", "have", "has", "had", \
                "having", "do", "does", "did", "doing", "a", "an", "the", "and", \
                "but", "if", "or", "because", "as", "until", "while", "of", "at", \
                "by", "for", "with", "about", "against", "between", "into", \
                "through", "during", "before", "after", "above", "below", "to", \
                "from", "up", "down", "in", "out", "on", "off", "over", "under", \
                "again", "further", "then", "once", "here", "there", "when", "where", \
                "why", "how", "all", "any", "both", "each", "few", "more", "most", \
                "other", "some", "such", "no", "nor", "not", "only", "own", "same", \
                "so", "than", "too", "very", "s", "t", "can", "will", "just", \
                "should", "", "best", "top", "unbelievable", "see", "xa", \
                "br", "ul", "li"}

bc = BertClient(ip='127.0.0.1', port=5555)

def evaluate(search_phrase, intent1, intent2, intent3):

    intent_list = [intent1, intent2, intent3]

    query = re.sub(' ','+',search_phrase)
    results = requests.get(f"https://www.google.com/search?client=firefox-b-1-d&q={query}&start=10")

    soup = BeautifulSoup(results.text, 'html.parser')

    def clean_phrases(list_of_contents, stopwords):
        cleaned_tokens = []
        for phrase in list_of_contents:
            if phrase.string:
                for word in re.finditer(r'([a-z]+)\W', phrase.string.lower()):
                    if word.group(1) not in stopwords:
                        cleaned_tokens.append(word.group(1))
        return cleaned_tokens

    article_titles = []
    h3s = soup.find_all('h3', 'r')
    for h3 in h3s:
        article_titles.append(clean_phrases(h3.a.contents, stopwords))

    article_contents = []
    meta_spans = soup.find_all('span', 'st')
    for span in meta_spans:
        article_contents.append(clean_phrases(span.contents, stopwords))

    # Join the titles and headers to prepare for BERT encoding
    complete_articles = [" ".join(set(title + header)) for title, header in zip(article_titles, article_contents)]

    # Get the encodings
    article_encodings = bc.encode(complete_articles)
    intent_encodings = bc.encode(intent_list)

    def get_valid_indices(encodings, sum_max = -7):
        nonzero_indeces_tuples = []
        for i, encoded_row in enumerate(encodings):
            for j, encoding in enumerate(encoded_row):
                if j != 0 and np.sum(encoding) < sum_max:
                    nonzero_indeces_tuples.append(tuple([i,j]))
        return nonzero_indeces_tuples

    valid_article_words = get_valid_indices(article_encodings, sum_max = -7)
    valid_intent_words = get_valid_indices(intent_encodings, sum_max = -7)

    comparisons_to_make = product(valid_intent_words, valid_article_words)

    similarities_by_intent = {0:defaultdict(lambda: []),1:defaultdict(lambda: list()),2:defaultdict(lambda: list())}

    for idxs in comparisons_to_make:
        intent_id = idxs[0][0]
        article_id = idxs[1][0]
        intent_vector = intent_encodings[idxs[0][0]][idxs[0][1]]
        article_vector = article_encodings[idxs[1][0]][idxs[1][1]]
        cosine = np.dot(article_vector, intent_vector, out=None)/(norm(article_vector)*norm(intent_vector))
        similarity_score = np.exp(cosine)
        similarities_by_intent[intent_id][article_id] += [np.round(similarity_score, 3)]

    article_means_by_intent = {0: [], 1: [], 2:[]}

    for intent_id, article_dict in similarities_by_intent.items():
        for article in article_dict:
            article_means_by_intent[intent_id].append(np.mean(article))


    results = {'search_phrase':search_phrase, 'intent_1': intent1, \
                'intent_2': intent2, 'intent_3': intent3, 'similarities':article_means_by_intent}

    return json.dumps(results)
