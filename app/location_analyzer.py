#!/usr/bin/env python
# coding: utf-8

import googlemaps
import re
from sklearn.cluster import KMeans
from bert_serving.client import BertClient
import requests
import json
from collections import defaultdict, Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import os
import socket


def evaluate(CHAIN_NAME='starbucks', CITY_NAME='austin', RADIUS='50000'):
    NEGATIVE_EMPHASIS = 2
    N_CLUSTERS_PER_SENTI = 3
    NUM_BERT_PODS = 3

    gmaps_key = os.environ['GMAPS_API_KEY']
    try:
        bert_ip = socket.gethostbyname('bert.default')
        print(f"Found bert IP: {bert_ip}")
    except:
        bert_ip = '127.0.0.1'
        print(f"Did not find bert IP via DNS lookup. Defaulted to: {bert_ip}")
    print(f"Trying to connect with Bert Server...")
    bc = BertClient(ip=bert_ip, port=5555)
    print(f"Established connection with Bert Server.")
    print(f"Loading GMaps API Key.")
    gm = googlemaps.Client(key=gmaps_key)
    print(f"Loaded GMaps API Key.")
    # ### Find the city's coordinates, then search for matching location names within a radius

    city_search = gm.find_place(input=CITY_NAME, input_type='textquery', \
                        fields=['place_id'])
    city_id = city_search['candidates'][0]['place_id']
    city_details = gm.place(place_id=city_id, fields=['name','formatted_address',\
                        'geometry'])
    city_lat = city_details['result']['geometry']['location']['lat']
    city_long = city_details['result']['geometry']['location']['lng']
    search_results = gm.places_nearby(location=(city_lat,city_long), radius=RADIUS, \
                        keyword=CHAIN_NAME)

    print(f"Found {len(search_results['results'])} locations.")


    # ### Isolate their Place IDs and search and fetch their reviews (currently top 20 locations)

    loc_ids = list()
    for location in search_results['results']:
        loc_ids.append(location['place_id'])

    loc_details = list()
    for loc_id in loc_ids:
        try:
            loc_details.append(gm.place(place_id=loc_id, fields=["name", "formatted_address", "rating", "review"]))
        except:
            print(f"could not find place for location id: {loc_id}")


    # ### Split the retrieved reviews by sentence, clean them, and prep them for encoding

    seg_rev_list = []
    seg_text = []
    loc_addresses = []
    seg_map_to_rev_and_loc = []

    rev_text = []
    rev_num = 0

    for loc_idx, location_details in enumerate(loc_details):
        loc_addresses.append(location_details['result']['formatted_address'])
        for review in location_details['result']['reviews']:
            this_review = review['text']
            if this_review != "":
                rev_text.append(this_review)
                # Creating (review_segment, review_id) tuples for each segment in the reviews
                for review_segment in re.findall(r"\w[\w’', %/:-?]+(?:.m.)?[\w’'% ,/:-]*", this_review):
                    if len(review_segment.strip()) < 2:
                        continue
                    # Counting if it's more than 20 tokens and splitting further if so
                    tokenized_review_segment = review_segment.split(' ')
                    if len(tokenized_review_segment) < 20:
                        seg_text.append(review_segment.strip())
                        seg_rev_list.append(review_segment.lower().strip())
                        seg_map_to_rev_and_loc.append((rev_num,loc_idx))
                    else:
                        while len(tokenized_review_segment) >= 20:
                            review_start = " ".join(tokenized_review_segment[:20])
                            seg_text.append(review_start.strip())
                            seg_rev_list.append(review_start.lower().strip())
                            seg_map_to_rev_and_loc.append((rev_num,loc_idx))
                            tokenized_review_segment = tokenized_review_segment[20:]
                rev_num += 1

    print(f"Requesting embeddings of {len(seg_rev_list)} review segments.")

    # ### Get the BERT embeddings
    
    seg_encodings = bc.encode(seg_rev_list, show_tokens=False)

    # ### Get sentiment embeddings

    print(f"Accumulating sentiment of {len(seg_rev_list)} review segments and locations.")

    sentibot = SentimentIntensityAnalyzer()

    seg_senti = []
    rev_dict_cumm_senti = defaultdict(lambda: (float(), [])) #float for accumulated sentiment, list for seg indices
    loc_dict_cumm_senti = defaultdict(lambda: (float(), []))

    for i, segmented_review in enumerate(seg_rev_list):
        senti_result = sentibot.polarity_scores(segmented_review)
        this_senti =  senti_result['pos'] - senti_result['neg'] * NEGATIVE_EMPHASIS
        rev_senti_so_far, rev_indices_so_far = rev_dict_cumm_senti[seg_map_to_rev_and_loc[i][0]]
        rev_dict_cumm_senti[seg_map_to_rev_and_loc[i][0]] = (round(rev_senti_so_far + this_senti, 1), rev_indices_so_far + [i])
        loc_senti_so_far, loc_indices_so_far = loc_dict_cumm_senti[seg_map_to_rev_and_loc[i][1]]
        loc_dict_cumm_senti[seg_map_to_rev_and_loc[i][1]] = (round(loc_senti_so_far + this_senti, 1), loc_indices_so_far + [i])

    positive_indices = []
    neutral_indices = []
    negative_indices = []

    for k, v in rev_dict_cumm_senti.items():
        cumm_senti, this_indices = v
        if cumm_senti < 0:
            negative_indices += this_indices
        elif cumm_senti < 1:
            neutral_indices += this_indices
        elif cumm_senti >= 1:
            positive_indices += this_indices

    # ### Concatenate sentiment embeddings with the BERT embeddings

    # print(f"Combining embeddings and sentiment...")

    # aug_encodings = []

    # current_idx = (None, None)
    # current_sentiment = [0] * 3
    # current_bert = [0] * 768
    # current_bert_norm = 0
    # new_lists_index = 0

    # sent_scores_by_list_of_ids_idx = defaultdict(int)

    # positive_indices = []
    # neutral_indices = []
    # negative_indices = []

    # for i, encoding in enumerate(seg_encodings):
    #     if seg_map_to_rev_and_loc[i] == current_idx:
    #         # We are continuing with more parts of a review, so we add the sentiment
    #         senti = seg_senti[i]
    #         current_sentiment = np.add(current_sentiment, np.multiply([senti['pos'], senti['neu'], senti['neg']], SENTIMENT_EMPHASIS))
    #         # We want the topic encoding with the largest magnitude from each review.
    #         this_norm = np.linalg.norm(encoding)
    #         if this_norm > current_bert_norm:
    #             current_bert_norm = this_norm
    #             current_bert = encoding
    #         # Comment out above and uncomment below to have cummulative topic encodings
    #         #current_bert = np.add(current_bert, encoding)
    #     else:
    #         # We are dealing with a new topic, so we add the previous topic.
    #         if current_bert_norm > 0:
    #             aug_encodings.append(np.append(current_bert,current_sentiment))

    #             # Add to the score accumulation by location
    #             cumm_senti_rating = round(current_sentiment[0] - current_sentiment[2],2)
    #             sent_scores_by_list_of_ids_idx[seg_map_to_rev_and_loc[i][1]] = round(sent_scores_by_list_of_ids_idx[seg_map_to_rev_and_loc[i][1]] + cumm_senti_rating, 1)

    #             #Sort this index by sentiment.
    #             if cumm_senti_rating < -3:
    #                 negative_indices.append(new_lists_index)
    #             elif cumm_senti_rating < 3:
    #                 neutral_indices.append(new_lists_index)
    #             elif cumm_senti_rating >= 3:
    #                 positive_indices.append(new_lists_index)

    #             new_lists_index += 1

    #         # Reset our per-review scores to this one.
    #         current_bert_norm = np.linalg.norm(encoding)
    #         current_bert = encoding
    #         senti = seg_senti[i]
    #         current_sentiment = np.multiply([senti['pos'], senti['neu'], senti['neg']], SENTIMENT_EMPHASIS)
    #         current_idx = seg_map_to_rev_and_loc[i]

    # Repeat this one last time for the last encoding
    # aug_encodings.append(np.append(current_bert,current_sentiment))
    # if cumm_senti_rating < -3:
    #     negative_indices.append(new_lists_index)
    # elif cumm_senti_rating < 3:
    #     neutral_indices.append(new_lists_index)
    # elif cumm_senti_rating >= 3:
    #     positive_indices.append(new_lists_index)

    def cluster_from_indices(indices_list, input_encodings = seg_encodings, input_text =seg_text, n_clusters=N_CLUSTERS_PER_SENTI):
        encodings = []
        text = []

        for idx in indices_list:
            encodings.append(input_encodings[idx])
            text.append(input_text[idx])

        if len(encodings) < 3:
            n_clusters = len(encodings)

        km = KMeans(n_clusters=n_clusters, max_iter=2400)
        seg_labels = km.fit_predict(encodings)
        seg_distances_to_all_ks = km.transform(encodings)
        seg_distance_to_nearest_k = []
        for i, label in enumerate(seg_labels):
            seg_distance_to_nearest_k.append(seg_distances_to_all_ks[i,label])

        return seg_labels, seg_distance_to_nearest_k, text

    pos_clust_labels, pos_clust_dist, pos_clust_text = cluster_from_indices(positive_indices)
    neu_clust_labels, neu_clust_dist, neu_clust_text = cluster_from_indices(neutral_indices)
    neg_clust_labels, neg_clust_dist, neg_clust_text = cluster_from_indices(negative_indices)

    # ### Identify Most Common Clustering Results

    stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                "you", "your", "yours", "yourself", "yourselves", "he", "him",
                "his", "himself", "she", "her", "hers", "herself", "it", "its",
                "itself", "they", "them", "their", "theirs", "themselves", "what",
                "which", "who", "whom", "this", "that", "these", "those", "am", "is",
                "are", "was", "were", "be", "been", "being", "have", "has", "had",
                "having", "do", "does", "did", "doing", "a", "an", "the", "and",
                "but", "if", "or", "because", "as", "until", "while", "of", "at",
                "by", "for", "with", "about", "against", "between", "into",
                "through", "during", "before", "after", "above", "below", "to",
                "from", "up", "down", "in", "out", "on", "off", "over", "under",
                "again", "further", "then", "once", "here", "there", "when", "where",
                "why", "how", "all", "any", "both", "each", "few", "more", "most",
                "other", "some", "such", "no", "nor", "not", "only", "own", "same",
                "so", "than", "too", "very", "didn", "s", "t", "can", "will", "just",
                "should", "", "best", "top", "unbelievable", "see", "xa", "br",
                "ul", "li", ".", "it's", "m", "re", "ve", "d", CHAIN_NAME.lower()}

    def get_most_common_words_per_cluster(cluster_labels, cluster_text, stop_ws = stopwords):

        cluster_word_counters = defaultdict(Counter)

        for idx, this_review in enumerate(cluster_text):
            for token in re.findall(r"[\w]+", this_review):
                if token.lower() not in stop_ws:
                    cluster_word_counters[cluster_labels[idx]][token] += 1
        
        return [v_counter.most_common(3) for k, v_counter in cluster_word_counters.items()]

    pos_most_common = get_most_common_words_per_cluster(pos_clust_labels, pos_clust_text)
    neu_most_common = get_most_common_words_per_cluster(neu_clust_labels, neu_clust_text)
    neg_most_common = get_most_common_words_per_cluster(neg_clust_labels, neg_clust_text)

    def package_for_return(labels, seg_text, rev_text, most_common_words, clust_dist, indices):

        cluster_dict = defaultdict(dict)

        full_text = []
        loc_ids = []
        rev_nums = []

        used_reviews = set()

        for i, seg in enumerate(seg_text):
            original_seg_index = indices[i]
            rev_num, loc_idx = seg_map_to_rev_and_loc[original_seg_index]
            full_text.append(rev_text[rev_num])
            loc_ids.append(loc_idx)
            rev_nums.append(rev_num)

        sort_order = np.argsort(np.array(clust_dist))

        sorted_labels = list(np.array(labels)[sort_order])
        sorted_seg_text = list(np.array(seg_text)[sort_order])
        sorted_full_text = list(np.array(full_text)[sort_order])
        sorted_loc_idx = list(np.array(loc_ids)[sort_order])
        sorted_rev_num = list(np.array(rev_nums)[sort_order])

        for idx, word_list in enumerate(most_common_words):
            cluster_dict[idx]['most_common_words'] = word_list
            cluster_dict[idx]['cluster_reviews'] = list()

        for idx, label in enumerate(sorted_labels):
            if sorted_rev_num[idx] not in used_reviews:
                used_reviews.add(sorted_rev_num[idx])
                match_string = r'(' + re.escape(sorted_seg_text[idx]) + r')'
                new_text = re.sub(match_string, r'<strong>\1</strong>', sorted_full_text[idx])
                cluster_dict[label]['cluster_reviews'].append(new_text + "&quot; - Said about Location #" + str(sorted_loc_idx[idx] + 1))

        return cluster_dict

        # {cluster_id:
        #     {'most_common_words': [list,of,words],
        #      'cluster_reviews': [list,of,review,texts]
        #     }
        # }

    pos_clusters = package_for_return(pos_clust_labels, pos_clust_text, rev_text, pos_most_common, pos_clust_dist, positive_indices)
    neu_clusters = package_for_return(neu_clust_labels, neu_clust_text, rev_text, neu_most_common, neu_clust_dist, neutral_indices)
    neg_clusters = package_for_return(neg_clust_labels, neg_clust_text, rev_text, neg_most_common, neg_clust_dist, negative_indices)

    #     review_clusters_top3 = dict()

    #     # for idx, label in enumerate(cluster_labels):
    #     #     sentiment = round(aug_encodings[idx][-3] - aug_encodings[idx][-1], 1)
    #     #     review_clusters[label].append((review_text[idx], sentiment)) #adding review text to each cluster



    # for cluster_no, cluster_list in review_clusters.items():
    #     cluster_total_senti = 0
    #     word_counter = Counter()
    #     for review_seg, this_senti in cluster_list:
    #         cluster_total_senti += this_senti
    #         tokens = review_seg.lower().split(' ')
    #         for token in tokens:
    #             if token not in stopwords:
    #                 word_counter[token] += 1
    #     avg_senti = cluster_total_senti / len(cluster_list)
    #     if avg_senti < -5:
    #         title_mod = "Negative"
    #     elif avg_senti < 5:
    #         title_mod = "Neutral"
    #     elif avg_senti >= 5:
    #         title_mod = "Positive"
    #     review_clusters_top3[cluster_no] = (title_mod, [x[0] for x in word_counter.most_common(3)])

    # ### Ranked positivity score by location

    sentiment_ranked_locations = sorted(loc_dict_cumm_senti.items(), key=lambda x: -x[1][0])

    # ### Packaging things up to be returned

    return_package = {'city_name': CITY_NAME, 'chain_name': CHAIN_NAME,
                      'location_addresses': loc_addresses,
                      'location_ranks_and_scores':sentiment_ranked_locations,
                      'pos_clusters':pos_clusters,
                      'neu_clusters':neu_clusters,
                      'neg_clusters':neg_clusters,
                      }

    return return_package
