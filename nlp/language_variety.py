# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os


import numpy as np
from dl_architecture import make_charvec, build_model, build_general_model
from sklearn.preprocessing import Normalizer
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn import preprocessing
from sklearn.metrics import f1_score

import gc
import string
import math

from collections import defaultdict
from keras import backend as K
from keras.utils.conv_utils import convert_kernel



def remove_email(text, replace_token):
    return re.sub(r'[\w\.-]+@[\w\.-]+', replace_token, text)


def remove_url(text, replace_token):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


def preprocess_data(df_data, column, tags_to_idx):

    #print("Data shape before preprocessing:", df_data.shape)
    df_data['text_clean'] = df_data[column].map(lambda x: remove_url(x, "HTTPURL"))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_email(x, 'EMAIL'))

    #print("Tags to idx: ", tags_to_idx)
    #print("Preprocessed data columns: ", df_data.columns)

    return df_data


class text_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]



def predict(data_test, column, lang, weights_path, data_path):

    textmodel_data = pickle.load(open(data_path, 'rb'))

    unigrams_shape, num_classes, charvec_shape, char_vocab_size, feature_union, char_vocab, max_train_len_char, tags_to_idx = textmodel_data
    #print(tags_to_idx)
    xtest = preprocess_data(data_test, column, tags_to_idx=tags_to_idx)
    feature_union_test = feature_union.transform(xtest)
    #print(unigrams_shape, num_classes, charvec_shape, char_vocab_size, feature_union_test.shape, len(char_vocab), max_train_len_char, tags_to_idx)

    charvec_test, _, _ = make_charvec(xtest.text_clean.tolist(), train=False, char_vocab=char_vocab, max_text_len=max_train_len_char)
    #print(charvec_test.shape)


   

    if lang != 'all':
        model = build_model(unigrams_shape, num_classes, charvec_shape, char_vocab_size)
    else:
        model = build_general_model(unigrams_shape, num_classes, charvec_shape, char_vocab_size)
    model.load_weights(weights_path)
    predictions = model.predict([feature_union_test, charvec_test]).argmax(axis=-1)
    return predictions, tags_to_idx
    