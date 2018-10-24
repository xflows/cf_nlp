import pandas as pd
#import os
from nltk.tag import pos_tag
from nltk.tokenize.treebank import TreebankWordTokenizer

def load_corpus_from_csv(input_dict):
    import gc
    separator = str(input_dict['separator'])
    if separator.startswith('\\'):
        separator = '\t'
    try:
        data_iterator = pd.read_csv(input_dict['file'], delimiter=separator, chunksize=1000, index_col=None, encoding = 'utf8')
        df_data = pd.DataFrame()
        for sub_data in data_iterator:
            df_data = pd.concat([df_data, sub_data], axis=0)
            gc.collect()
    except:
        raise Exception("Ups, we are having problem uploading your corpus. Please make sure it's encoded in utf-8.")
    df_data = df_data.dropna()
    #print(df_data.columns.tolist())
    #print("Data shape:", df_data.shape)
    return {'dataframe': df_data}


def select_corpus_attribute(input_dict):
    df = input_dict['dataframe']
    attribute = input_dict['attribute']
    column = df[attribute].tolist()
    return {'attribute': column}


def nltk_tokenizer(input_dict):
    """Prejme seznam stringov, ki jih nato tokenizira z nltk-jevim vgrajenim tokenizatorjem
    Treebank (glej https://www.nltk.org/api/nltk.tokenize.html). Vrne seznam tokeniziranih
    povedi."""
    list_of_sentences = input_dict['attribute']
    tokenizer = TreebankWordTokenizer()
    tokens = []
    for sent in list_of_sentences:
        tokens.append(tokenizer.tokenize(sent))
    #print(tokens)
    return {'tokens': tokens}


def nltk_pos_tagger(input_dict):
    """Prejme dataframe z dvema stolpcema, ki vsebujeta izvirne povedi in tokenizirane povedi. Vrne
    dataframe s tremi stolpci, v tretjem stolpcu so shranjeni tagi za vsak stolpec. Uporabljen je
    priporočeni tagger iz knjižnice nltk."""
    df_tokenized_sentences = input_dict['tokens']
    tagged_sents = []
    for index, row in df_tokenized_sentences.iterrows():
        tagged_sents.append(pos_tag(row['tokens']))

    # razpakiranje terk (token,tag) v seznam s tagi
    for list in tagged_sents[:]:
        tag_list = []
        if list:
            for tag_tuple in list:
                tag_list.append(tag_tuple[1])
        tagged_sents.remove(list)
        tagged_sents.append(tag_list)

    # pakiranje seznamov s tagi v dataframe
    tagged_sents_series = pd.Series(tagged_sents, name='tags')
    df_tags = pd.DataFrame(tagged_sents_series)
    print(df_tags)
    df_tokenized_sentences = df_tokenized_sentences.join(df_tags)
    print(df_tokenized_sentences)
    return {'tagged_sents': df_tokenized_sentences}


def affix_extractor(input_dict):
    corpus = input_dict['corpus']
    affixes_tokens = []
    affix_type = input_dict['affix_type']
    affix_length = int(input_dict['affix_length'])
    punct = '#@!"$%&()*+,-./:;<=>?[\]^_`{|}~' + "'"
    for text in corpus:
        if affix_type == 'suffix':
            affixes = " ".join([word[-affix_length:] for word in text.split() if len(word) >= affix_length])
        elif affix_type == 'prefix':
            affixes = " ".join([word[0:affix_length] for word in text.split() if len(word) >= affix_length])
        else:
            ngrams = []
            for i, character in enumerate(text[0:-affix_length - 1]):
                ngram = text[i:i+affix_length]
                if ngram[0] in punct:
                    for p in punct:
                        if p in ngram[1:]:
                            break
                    else:
                       ngrams.append(ngram)
            affixes = "###".join(ngrams)
        affixes_tokens.append(affixes)
    return {'affixes': affixes_tokens}


def build_dataframe_from_columns(input_dict):
    columns = input_dict['corpus']
    names = [str(name).strip() for name in input_dict['names'].split(',')]
    if len(names) != len(columns):
        names = ['Column_' + str(i+1) for i in range(len(columns))]
    df = pd.DataFrame(columns)
    df = df.transpose()
    df.columns = names
    return {'df': df}
