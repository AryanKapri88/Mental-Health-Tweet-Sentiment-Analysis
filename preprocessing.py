#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from nltk.tokenize import word_tokenize
from emot.emo_unicode import UNICODE_EMOJI # for emojis
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words=set(stop_words)
print(stop_words)

emoji = list(UNICODE_EMOJI.keys())


def clean_tokens(text):
    # 1 create tokens
    tokens = word_tokenize(text)
    # 2 lower case
    tokens = [w.lower() for w in tokens]
    # 3 remove punctuations
    stripped = [word for word in tokens if word.isalpha()]
    # 4 remove stop_words
    stop_words = set(stopwords.words('english'))
    words = [w for w in stripped if not w in stop_words]
    # remove emojis
    no_emoji = [w for w in words if not w in emoji]
    # join cleaned tokens into a single string
    clean_text = ' '.join(no_emoji)
    # return the cleaned string
    return clean_text


# lemmatization of word, changing words into their roots words
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()
    def __call__(self,tweet):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(tweet)]

