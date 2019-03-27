import requests
import re
from bs4 import BeautifulSoup
from collections import defaultdict
import numpy as np

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
                "should", ""}

def evaluate(SEARCH_PHRASE):

    query = re.sub(' ','+',SEARCH_PHRASE)
    results = requests.get(f"https://www.google.com/search?q={query}&start=10")

    soup = BeautifulSoup(results.text, 'html.parser')
    pattern = re.compile('[\W_]+ ', re.UNICODE)

    article_titles = []
    h3s = soup.find_all('h3', 'r')
    for h3 in h3s:
        cleaned_title = ""
        phrases = h3.find('a').contents
        for phrase in phrases:
            phrase = re.sub(r'</?b>',' ',str(phrase))
            cleaned_title = cleaned_title + pattern.sub(' ', str(phrase))

        article_titles.append(cleaned_title)

    article_contents = []
    spans = soup.find_all('span', 'st')
    for span in spans:
        cleaned_article = ""
        for phrase in span.contents:
            phrase = re.sub(r'</?b>',' ',str(phrase))
            phrase = re.sub(r'<br/>',' ',phrase)
            phrase = re.sub(r'\n',' ',phrase)
            phrase = re.sub(r'\xa0',' ',phrase)
            cleaned_article = cleaned_article + pattern.sub(' ', str(phrase))
        article_contents.append(cleaned_article)

    articles = zip(article_titles, article_contents)

    word_counter = defaultdict(int)

    for i, article in enumerate(article_contents):
        content = article_titles[0] + " " + article_contents[i]
        for token in content.split(" "):
            t = token.lower()
            word_counter[t] = word_counter[t] + 1

    counts = []
    tokens = []

    for k,v in word_counter.items():
        counts.append(v)
        tokens.append(k)

    sorted_indices = np.argsort(counts)

    query_results = {"search_phrase": SEARCH_PHRASE, "wordcounts": []}

    for index in reversed(sorted_indices):
        if tokens[index] not in stopwords:
            query_results['wordcounts'].append({"token":tokens[index].strip('\"'), "count":counts[index]})

    return query_results
