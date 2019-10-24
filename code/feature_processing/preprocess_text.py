#from gensim.models import Word2Vec
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def to_index_vectors(texts, max_words=250, vectorizer=None, unique_words=True, vocab_size=5000):

    if vectorizer is None:
        # Fit vectorizer if not passed in
        vectorizer = CountVectorizer(stop_words='english', binary=True, max_features=vocab_size)
        vectorizer.fit(texts)

    id_map = vectorizer.vocabulary_

    text_vectors = []
    for text in texts:

        text = text.replace('This measure has not been amended since it was introduced.', '')

        # TEMP: use set of words to get more representative sample
        if unique_words:
            words = set(text.split())
        else:
            words = text.split()
        # If word in vectorizer, get its index
        # Offset by 1 to allow for padding
        vec = [id_map[w] + 1 for w in words if w in id_map]

        # Add padding if necessary
        if len(vec) < max_words:
            vec = vec + [0] * (max_words - len(vec))

        # Only keep max words of the words
        # TODO: pick better subset here
        text_vectors.append(vec[:max_words])
    
    #print(text_vectors[:10])
    #print(sum([len(np.array(x).nonzero()[0]) for x in text_vectors]) / len(text_vectors), len(text_vectors))
    return text_vectors, vectorizer


def vectorize_text(bill_texts, max_words=250, vectorizer=None, unique_words=True, text_field='text', vocab_size=5000):

    clean_vote_text = {}

    # Store one copy of each bill, to avoid duplicate vectorization
    for bill_id, data in bill_texts.groupby('natural_id').first().iterrows():

            clean_vote_text[bill_id] = data[text_field]

    # Trick to preserve ordering
    bids, texts = zip(*clean_vote_text.items())

    text_vectors, vectorizer = to_index_vectors(texts, max_words=max_words, vectorizer=vectorizer, unique_words=unique_words, vocab_size=vocab_size)

    # Map text vector back to each training point
    vec_map = dict(zip(bids, text_vectors))
    bill_texts['text_feats'] = bill_texts.natural_id.map(vec_map)

    return(bill_texts, vectorizer)


def prepare_text_features(vote_df, max_words=250,  unique_words=True, text_field='special_text', vocab_size=5000, vectorizer=None):

    # Prepare vectorizer on data
    vote_df, vectorizer = vectorize_text(vote_df, max_words=max_words, 
                                         unique_words=unique_words, text_field=text_field, 
                                         vocab_size=vocab_size, vectorizer=vectorizer)
    
    return np.array(list(vote_df['text_feats'])), vectorizer
