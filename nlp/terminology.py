# -*- coding: utf-8 -*-
import pandas as pd
from nltk import word_tokenize
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
from collections import defaultdict
import editdistance
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn import pipeline
import math
import argparse

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
import os
from io import open as openio




#used for manual evaluation
def build_manual_eval_set(terms_src, terms_tar):
    all_terms = []
    for src_term in terms_src:
        for tar_term in terms_tar:
            all_terms.append([src_term.strip(), tar_term.strip()])
    df = pd.DataFrame(all_terms)
    df.columns = ['src_term', 'tar_term']
    return df


def lemmatize(text, lemmatizer):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(x) for x in tokens]
    return " ".join(tokens).lower()


def preprocess(text):
    tokens = word_tokenize(text)
    return " ".join(tokens).lower()



def longest_common_substring(s):
    s = s.split('\t')
    s1, s2 = s[0], s[1]
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def longest_common_subsequence(s):
    s = s.split('\t')
    a, b = s[0], s[1]
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = a[x - 1] + result
            x -= 1
            y -= 1
    return result


def isFirstWordTranslated(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    firstWordSource = term1.split()[0].strip()
    firstWordTarget = term2.split()[0].strip()
    for target, p in giza_dict[firstWordSource]:
        if target == firstWordTarget:
            return 1

    #fix for compounding problem
    if len(term2) > 4:
        for target, p in giza_dict[firstWordSource]:
            if term2.startswith(target):
                #print(term2, target)
                return 1
    return 0


def isLastWordTranslated(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    lastWordSource = term1.split()[-1].strip()
    lastWordTarget = term2.split()[-1].strip()
    for target, p in giza_dict[lastWordSource]:
        if target == lastWordTarget:
            return 1

    # fix for compounding problem
    if len(term2) > 4:
        for target, p in giza_dict[lastWordSource]:
            if term2.endswith(target):
                return 1
    return 0



def percentageOfTranslatedWords(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    for word in term1:
        for target, p in giza_dict[word]:
            if target in term2:
                counter+=1
                break
    return float(counter)/len(term1)


def percentageOfNotTranslatedWords(s, giza_dict):
    return 1 - percentageOfTranslatedWords(s, giza_dict)


def longestTranslatedUnitInPercentage(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    for word in term1:
        for target, p in giza_dict[word]:
            if target in term2:
                counter += 1
                if counter > max:
                    max = counter
                break
        else:
            counter = 0
    return float(max) / len(term1)


def longestNotTranslatedUnitInPercentage(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    for word in term1:
        for target, p in giza_dict[word]:
            if target in term2:
                counter = 0
                break
        else:
            counter += 1
            if counter > max:
                max = counter
    return float(max) / len(term1)


def isLemmatizedWordCovered(x, giza_dict, index):
    terms = x.split('\t')
    term_source_lemma, term_target_lemma = terms[0], terms[1]
    for word, score in giza_dict[term_source_lemma.split()[index]]:
        if word in term_target_lemma.split():
            return 1
    lcstr = float(len(longest_common_substring(term_source_lemma.split()[index] + '\t' + term_target_lemma))) / max(len(term_source_lemma.split()[index]), len(term_target_lemma))
    lcsr = float(len(longest_common_subsequence(term_source_lemma.split()[index] + '\t' + term_target_lemma))) / max(len(term_source_lemma.split()[index]), len(term_target_lemma))
    dice = 2 * float(len(longest_common_substring(term_source_lemma.split()[index] + '\t' + term_target_lemma))) / (len(term_source_lemma.split()[index]) + len(term_target_lemma))
    nwd = float(len(longest_common_substring(term_source_lemma.split()[index] + '\t' + term_target_lemma))) / min(len(term_source_lemma.split()[index]),len(term_target_lemma))
    editDistance = 1 - (float(editdistance.eval(term_source_lemma.split()[index], term_target_lemma)) / max(len(term_source_lemma.split()[index]), len(term_target_lemma)))
    if max(lcstr, lcsr, dice, nwd, editDistance) > 0.7:
        return 1
    return 0


def isWordCovered(x, giza_dict, index):
    terms = x.split('\t')
    term_source, term_target = terms[0], terms[1]
    for word, score in giza_dict[term_source.split()[index]]:
        if word in term_target.split():
            return 1
    lcstr = float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / max(
        len(term_source.split()[index]), len(term_target))
    lcsr = float(len(longest_common_subsequence(term_source.split()[index] + '\t' + term_target))) / max(
        len(term_source.split()[index]), len(term_target))
    dice = 2 * float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / (
    len(term_source.split()[index]) + len(term_target))
    nwd = float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / min(
        len(term_source.split()[index]), len(term_target))
    editDistance = 1 - (float(editdistance.eval(term_source.split()[index], term_target)) / max(
        len(term_source.split()[index]), len(term_target)))
    if max(lcstr, lcsr, dice, nwd, editDistance) > 0.7:
        return 1
    return 0


def percentageOfCoverage(x, giza_dict):
    terms = x.split('\t')
    length = len(terms[0].split())
    counter = 0
    for index in range(length):
        counter += isWordCovered(x, giza_dict, index)
    return counter/length


def preprocess(text):
    tokens = word_tokenize(text)
    return " ".join(tokens).lower()


def transcribe(text, lang):
    sl_repl = {'č':'ch', 'š':'sh', 'ž': 'zh'}
    en_repl = {'x':'ks', 'y':'j', 'w':'v', 'q':'k'}
    fr_repl = {'é':'e', 'à':'a', 'è':'e', 'ù':'u', 'â':'a', 'ê':'e', 'î':'i', 'ô':'o', 'û':'u', 'ç':'c', 'ë':'e', 'ï':'i', 'ü':'u'}
    if lang == 'en':
        en_tr = [en_repl.get(item,item)  for item in list(text)]
        return "".join(en_tr).lower()
    elif lang == 'sl':
        sl_tr = [sl_repl.get(item,item)  for item in list(text)]
        return "".join(sl_tr).lower()
    elif lang == 'fr':
        fr_tr = [fr_repl.get(item, item) for item in list(text)]
        return "".join(fr_tr).lower()
    else:
        print ('unknown language for transcription')



def createFeatures(data, giza_dict, giza_dict_reversed):
    data['src_term'] = data['src_term'].map(lambda x: preprocess(x))
    data['tar_term'] = data['tar_term'].map(lambda x: preprocess(x))
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']

    data['isFirstWordTranslated'] = data['term_pair'].map(lambda x: isFirstWordTranslated(x, giza_dict))
    data['isLastWordTranslated'] = data['term_pair'].map(lambda x: isLastWordTranslated(x, giza_dict))
    data['percentageOfTranslatedWords'] = data['term_pair'].map(
        lambda x: percentageOfTranslatedWords(x, giza_dict))
    data['percentageOfNotTranslatedWords'] = data['term_pair'].map(
        lambda x: percentageOfNotTranslatedWords(x, giza_dict))
    data['longestTranslatedUnitInPercentage'] = data['term_pair'].map(
        lambda x: longestTranslatedUnitInPercentage(x, giza_dict))
    data['longestNotTranslatedUnitInPercentage'] = data['term_pair'].map(
        lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict))

    data['term_pair'] = data['tar_term'] + '\t' + data['src_term']

    data['isFirstWordTranslated_reversed'] = data['term_pair'].map(
        lambda x: isFirstWordTranslated(x, giza_dict_reversed))
    data['isLastWordTranslated_reversed'] = data['term_pair'].map(
        lambda x: isLastWordTranslated(x, giza_dict_reversed))
    data['percentageOfTranslatedWords_reversed'] = data['term_pair'].map(
        lambda x: percentageOfTranslatedWords(x, giza_dict_reversed))
    data['percentageOfNotTranslatedWords_reversed'] = data['term_pair'].map(
        lambda x: percentageOfNotTranslatedWords(x, giza_dict_reversed))
    data['longestTranslatedUnitInPercentage_reversed'] = data['term_pair'].map(
        lambda x: longestTranslatedUnitInPercentage(x, giza_dict_reversed))
    data['longestNotTranslatedUnitInPercentage_reversed'] = data['term_pair'].map(
        lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict_reversed))

    data['src_term_tr'] = data['src_term'].map(lambda x: transcribe(x, 'en'))
    data['tar_term_tr'] = data['tar_term'].map(lambda x: transcribe(x, lang))
    data['term_pair_tr'] = data['src_term_tr'] + '\t' + data['tar_term_tr']
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']

    data['longestCommonSubstringRatio'] = data['term_pair_tr'].map(
        lambda x: float(len(longest_common_substring(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['longestCommonSubsequenceRatio'] = data['term_pair_tr'].map(
        lambda x: float(len(longest_common_subsequence(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['dice'] = data['term_pair_tr'].map(
        lambda x: (2 * float(len(longest_common_substring(x)))) / (len(x.split('\t')[0]) + len(x.split('\t')[1])))
    data['NWD'] = data['term_pair_tr'].map(
        lambda x: float(len(longest_common_substring(x))) / min(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['editDistanceNormalized'] = data['term_pair_tr'].map(lambda x: 1 - (
    float(editdistance.eval(x.split('\t')[0], x.split('\t')[1])) / max(len(x.split('\t')[0]), len(x.split('\t')[1]))))

    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']

    data['isFirstWordCovered'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict, 0))
    data['isLastWordCovered'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict, -1))
    data['percentageOfCoverage'] = data['term_pair'].map(lambda x: percentageOfCoverage(x, giza_dict))
    data['percentageOfNonCoverage'] = data['term_pair'].map(lambda x: 1 - percentageOfCoverage(x, giza_dict))
    data['diffBetweenCoverageAndNonCoverage'] = data['percentageOfCoverage'] - data['percentageOfNonCoverage']

    data['term_pair'] = data['tar_term'] + '\t' + data['src_term']

    data['isFirstWordCovered_reversed'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict_reversed, 0))
    data['isLastWordCovered_reversed'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict_reversed, -1))
    data['percentageOfCoverage_reversed'] = data['term_pair'].map(lambda x: percentageOfCoverage(x, giza_dict_reversed))
    data['percentageOfNonCoverage_reversed'] = data['term_pair'].map(lambda x: 1 - percentageOfCoverage(x, giza_dict_reversed))
    data['diffBetweenCoverageAndNonCoverage_reversed'] = data['percentageOfCoverage_reversed'] - data['percentageOfNonCoverage_reversed']

    data['averagePercentageOfTranslatedWords'] = (data['percentageOfTranslatedWords'] + data['percentageOfTranslatedWords_reversed']) / 2

    data = data.drop(['term_pair', 'term_pair_tr', 'src_term_tr', 'tar_term_tr'],axis=1)

    #print('feature construction done')
    return data



def createLemmatizedFeatures(data, giza_dict, giza_dict_reversed):
    lemmatizer_en = Lemmatizer(dictionary=lemmagen.DICTIONARY_ENGLISH)
    data['src_term_lemma'] = data['src_term'].map(lambda x: lemmatize(x, lemmatizer_en))
    lemmatizer_sl = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
    data['tar_term_lemma'] = data['tar_term'].map(lambda x: lemmatize(x, lemmatizer_sl))
    data['term_pair_lemma'] = data['src_term_lemma'] + '\t' + data['tar_term_lemma']

    data['isFirstWordTranslated'] = data['term_pair_lemma'].map(lambda x: isFirstWordTranslated(x, giza_dict))
    data['isLastWordTranslated'] = data['term_pair_lemma'].map(lambda x: isLastWordTranslated(x, giza_dict))
    data['percentageOfTranslatedWords'] = data['term_pair_lemma'].map(lambda x: percentageOfTranslatedWords(x, giza_dict))
    data['percentageOfNotTranslatedWords'] = data['term_pair_lemma'].map(lambda x: percentageOfNotTranslatedWords(x, giza_dict))
    data['longestTranslatedUnitInPercentage'] = data['term_pair_lemma'].map(lambda x: longestTranslatedUnitInPercentage(x, giza_dict))
    data['longestNotTranslatedUnitInPercentage'] = data['term_pair_lemma'].map(lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict))

    data['term_pair_lemma'] = data['tar_term_lemma'] + '\t' + data['src_term_lemma']

    data['isFirstWordTranslated_reversed'] = data['term_pair_lemma'].map(lambda x: isFirstWordTranslated(x, giza_dict_reversed))
    data['isLastWordTranslated_reversed'] = data['term_pair_lemma'].map(lambda x: isLastWordTranslated(x, giza_dict_reversed))
    data['percentageOfTranslatedWords_reversed'] = data['term_pair_lemma'].map(lambda x: percentageOfTranslatedWords(x, giza_dict_reversed))
    data['percentageOfNotTranslatedWords_reversed'] = data['term_pair_lemma'].map(lambda x: percentageOfNotTranslatedWords(x, giza_dict_reversed))
    data['longestTranslatedUnitInPercentage_reversed'] = data['term_pair_lemma'].map(lambda x: longestTranslatedUnitInPercentage(x, giza_dict_reversed))
    data['longestNotTranslatedUnitInPercentage_reversed'] = data['term_pair_lemma'].map(lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict_reversed))
    
    data['src_term_tr'] = data['src_term'].map(lambda x: transcribe(x, 'en'))
    data['tar_term_tr'] = data['tar_term'].map(lambda x: transcribe(x, 'sl'))
    data['term_pair_tr'] = data['src_term_tr'] + '\t' + data['tar_term_tr']
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']
    #print(data['term_pair_tr'])

    data['longestCommonSubstringRatio'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_substring(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['longestCommonSubsequenceRatio'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_subsequence(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['dice'] = data['term_pair_tr'].map(lambda x: (2 * float(len(longest_common_substring(x)))) / (len(x.split('\t')[0]) + len(x.split('\t')[1])))
    data['NWD'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_substring(x))) / min(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['editDistanceNormalized'] = data['term_pair_tr'].map(lambda x: 1 - (float(editdistance.eval(x.split('\t')[0], x.split('\t')[1])) / max(len(x.split('\t')[0]), len(x.split('\t')[1]))))

    data['term_pair_lemma'] = data['src_term_lemma'] + '\t' + data['tar_term_lemma']

    data['isFirstWordCovered'] = data['term_pair_lemma'].map(lambda x: isLemmatizedWordCovered(x, giza_dict, 0))
    data['isLastWordCovered'] = data['term_pair_lemma'].map(lambda x: isLemmatizedWordCovered(x, giza_dict, -1))
    data['percentageOfCoverage'] = data['term_pair_lemma'].map(lambda x: percentageOfCoverage(x, giza_dict))
    data['percentageOfNonCoverage'] = data['term_pair_lemma'].map(lambda x: 1 -percentageOfCoverage(x, giza_dict))
    data['diffBetweenCoverageAndNonCoverage'] = data['percentageOfCoverage'] - data['percentageOfNonCoverage']

    data['term_pair_lemma'] = data['tar_term_lemma'] + '\t' + data['src_term_lemma']

    data['isFirstWordCovered_reversed'] = data['term_pair_lemma'].map(lambda x: isLemmatizedWordCovered(x, giza_dict_reversed, 0))
    data['isLastWordCovered_reversed'] = data['term_pair_lemma'].map(lambda x: isLemmatizedWordCovered(x, giza_dict_reversed, -1))
    data['percentageOfCoverage_reversed'] = data['term_pair_lemma'].map(lambda x: percentageOfCoverage(x, giza_dict_reversed))
    data['percentageOfNonCoverage_reversed'] = data['term_pair_lemma'].map(lambda x: 1 - percentageOfCoverage(x, giza_dict_reversed))
    data['diffBetweenCoverageAndNonCoverage_reversed'] = data['percentageOfCoverage_reversed'] - data['percentageOfNonCoverage_reversed']

    data['averagePercentageOfTranslatedWords'] = (data['percentageOfTranslatedWords'] + data['percentageOfTranslatedWords_reversed']) / 2


    data = data.drop(['term_pair', 'term_pair_lemma', 'src_term_lemma', 'tar_term_lemma', 'term_pair_tr', 'src_term_tr', 'tar_term_tr'], axis = 1)

    #print('feature construction done')
    return data


def arrangeLemmatizedData(input, lemmatization=False, reverse=False):
    dd = defaultdict(list)
    with openio(input, encoding='utf8') as f:
        for line in f:
            line = line.split()
            source, target, score = line[0], line[1], line[2]
            source = source.strip('`’“„,‘')
            target = target.strip('`’“„,‘')
            if lemmatization and not reverse:
                lemmatizer_en = Lemmatizer(dictionary=lemmagen.DICTIONARY_ENGLISH)
                source = lemmatizer_en.lemmatize(source)
                lemmatizer_sl = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
                target= lemmatizer_sl.lemmatize(target)
            elif lemmatization and reverse:
                lemmatizer_sl = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
                source = lemmatizer_sl.lemmatize(source)
                lemmatizer_en = Lemmatizer(dictionary=lemmagen.DICTIONARY_ENGLISH)
                target = lemmatizer_en.lemmatize(target)

            dd[source].append((target, score))

    for k, v in dd.items():
        v = sorted(v, key=lambda tup: float(tup[1]), reverse = True)
        new_v = []
        for word, p in v:
            if (len(k) < 4 and len(word) > 5) or (len(word) < 4 and len(k) > 5):
                continue
            if float(p) < 0.05:
                continue
            new_v.append((word, p))
        dd[k] = new_v
    return dd


def arrangeData(input):
    dd = defaultdict(list)
    with openio(input, encoding='utf8') as f:
        for line in f:
            try:
                source, target, score = line.split()
                source = source.strip('`’“„,‘')
                target = target.strip('`’“„,‘')
                dd[source].append((target, score))
            except:
                pass
                #print(line)

    for k, v in dd.items():
        v = sorted(v, key=lambda tup: float(tup[1]), reverse=True)
        new_v = []
        for word, p in v:
            if (len(k) < 4 and len(word) > 5) or (len(word) < 4 and len(k) > 5):
                continue
            if float(p) < 0.05:
                continue
            new_v.append((word, p))
        dd[k] = new_v
    return dd


def filterTrainSet(df, ratio):
    df_pos = df[df['label'] == 1]
    df_pos = df_pos[df_pos['isFirstWordTranslated'] == 1]
    df_pos = df_pos[df_pos['isLastWordTranslated'] == 1]
    df_pos = df_pos[df_pos['isFirstWordTranslated_reversed'] == 1]
    df_pos = df_pos[df_pos['isLastWordTranslated_reversed'] == 1]
    df_pos = df_pos[df_pos['percentageOfCoverage'] > 0.66]
    df_pos = df_pos[df_pos['percentageOfCoverage_reversed'] > 0.66]

    df_neg = df[df['label'] == 0].sample(frac=1, random_state=123)[:df_pos.shape[0] * ratio]
    df = pd.concat([df_pos, df_neg])
    return df



class digit_col(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        hd_searches = hd_searches.drop(['src_term', 'tar_term'], axis=1)
        return hd_searches.values






















