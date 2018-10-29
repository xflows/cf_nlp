import pandas as pd
import os
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
    """Prejme seznam tokeniziranih povedi. Vrne seznam tagov za vsako tokenizirano poved.
    Uporabljen je priporočeni tagger iz knjižnice nltk."""
    tokenized_sentences = input_dict['tokens']
    tagged_sents = []
    for sentence in tokenized_sentences:
        tagged_sents.append(pos_tag(sentence))

    # razpakiranje terk (token,tag) v seznam s tagi
    for list in tagged_sents[:]:
        tag_list = []
        if list:
            for tag_tuple in list:
                tag_list.append(tag_tuple[1])
        tagged_sents.remove(list)
        tagged_sents.append(tag_list)

    #print(tagged_sents)
    return {'tagged_sents': tagged_sents}


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


def concatenate_corpora(input_dict):
    dfs = input_dict['dfs']
    return {'df': pd.concat(dfs)}


def count_patterns(input_dict):
    from itertools import groupby
    corpus = input_dict['corpus']
    mode = input_dict['mode']
    wordlist = input_dict['custom'].split(',')
    wordlist = [word.strip() for word in wordlist]
    sum_all = input_dict['sum_all']
    raw_frequency = input_dict['raw_frequency']
    if mode == 'emojis':
        folder_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(folder_path, 'models', 'emoji_dataset.csv')
        df_emojis = pd.read_csv(path, encoding="utf-8", delimiter=",")
        emoji_list = set(df_emojis['Emoji'].tolist())
    counts = []
    whole_length = 0
    for doc in corpus:
        doc_length = len(doc)
        if doc_length == 0:
            counts.append(0)
            continue
        cnt = 0
        if mode == 'floods':
            text = ''.join(doc.split())
            groups = groupby(text)
            for label, group in groups:
                char_cnt = sum(1 for _ in group)
                if char_cnt > 2:
                    cnt += 1
        elif mode == 'emojis':
            for emoji in emoji_list:
                cnt += doc.count(emoji)
        else:
            for word in wordlist:
                cnt += doc.count(word)
        counts.append(float(cnt)/doc_length) if not raw_frequency and not sum_all else counts.append(cnt)
        whole_length += doc_length
    if not sum_all:
        return {'counts': counts}
    else:
        if raw_frequency:
            return {'counts': sum(counts)}
        return {'counts': float(sum(counts))/whole_length}


def emoji_sentiment(input_dict):
    corpus = input_dict['corpus']
    emoji_dict = {}
    folder_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(folder_path, 'models', 'emoji_dataset.csv')
    df_emojis = pd.read_csv(path, delimiter=",", encoding="utf-8")
    for index, row in df_emojis.iterrows():
        occurrences = float(row['Occurrences'])
        pos = (float(row['Positive']) + 1) / (occurrences + 3)
        neg = (float(row['Negative']) + 1) / (occurrences + 3)
        sent = pos - neg
        emoji_dict[row['Emoji']] = sent
    sentiments = []
    for doc in corpus:
        sentiment = 0
        l = emoji_dict.keys()
        for pattern in l:
            text_cnt = doc.count(pattern)
            sentiment += float(emoji_dict[pattern]) * text_cnt
        sentiments.append(sentiment)
    return {'sentiments': sentiments}


def extract_true_and_predicted_labels(input_dict):
    df = input_dict['dataframe']
    true_values = input_dict['true_values']
    predicted_values = input_dict['predicted_values']
    true_values = df[true_values].tolist()
    predicted_values = df[predicted_values].tolist()
    return {'labels': [true_values, predicted_values]}


def filter_corpus(input_dict):
    corpus = input_dict['dataframe']
    query = input_dict['query']
    if '>' in query:
        query = query.split('>')
        column_name, value = query[0].strip(), float(query[1].strip())
        corpus = corpus[corpus[column_name] > value]
    elif '<' in query:
        query = query.split('<')
        column_name, value = query[0].strip(), float(query[1].strip())
        corpus = corpus[corpus[column_name] < value]
    elif '==' in query:
        query = query.split('==')
        column_name, value = query[0].strip(), query[1].strip()
        corpus = corpus[corpus[column_name] == value]
    elif '!=' in query:
        query = query.split('!=')
        column_name, value = query[0].strip(), query[1].strip()
        corpus = corpus[corpus[column_name] != value]
    elif 'in' in query:
        query = query.split(' in ', 1)
        value, column_name = query[0].strip(), query[1].strip()
        corpus = corpus[corpus[column_name].str.contains(value)]
    return {'dataframe': corpus}


def group_by_column(input_dict):
    chosen_column = input_dict['column']
    df = input_dict['df']
    columns = df.columns.tolist()
    #print(columns)
    columns.remove(chosen_column)
    group_dict = {}
    for index, row in df.iterrows():
        if row[chosen_column] not in group_dict:
            chosen_column_dict = {}
            for column in columns:
                chosen_column_dict[column] = [row[column]]
        else:
            chosen_column_dict = group_dict[row[chosen_column]]
            for column in columns:
                chosen_column_dict[column].append(row[column])
        group_dict[row[chosen_column]] = chosen_column_dict
    df_list = []
    for key, value in group_dict.items():
        end_dict = {}
        end_dict[chosen_column] = key
        for column in columns:
            end_dict[column] = " ".join([str(x) for x in value[column]]).replace('\n', ' ')
        df_list.append(end_dict)
    df_grouped = pd.DataFrame(df_list)
    return {'df': df_grouped}

#def gender_classification(input_dict):
#    from gender_classification import preprocess, createFeatures, simplify_tag
#    lang = input_dict['lang']
#    df = input_dict['dataframe']
#    column = input_dict['column']
#    output_name = input_dict['output_name']
#    corpus = df[column].tolist()
#    folder_path = os.path.dirname(os.path.realpath(__file__))
#    path = os.path.join(folder_path, 'models', 'gender_classification', 'lr_clf_' + lang + '_gender_python2.pkl')
#    sys.modules['gender_classification'] = genclass

    # get pos tags
#    if lang == 'en':
#        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#        pos_tags = PerceptronTagger()
#    else:
#        pos_tags = PerceptronTagger(load=False)
#        if lang == 'es':
#            sent_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
#            pos_tags.train(list(cess.tagged_sents()))
#        elif lang == 'pt':
#            sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
#            tsents = floresta.tagged_sents()
#            tsents = [[(w.lower(), simplify_tag(t)) for (w, t) in sent] for sent in tsents if sent]
#            pos_tags.train(tsents)
#        else:
#            sent_tokenizer = None

#    df_data = pd.DataFrame({'text': corpus})

#    df_prep = preprocess(df_data, lang, pos_tags, sent_tokenizer)
#    df_data = createFeatures(df_prep)

#    X = df_data

#    clf = joblib.load(path)
#    y_pred_gender = clf.predict(X)

#    df_results = pd.DataFrame({output_name: y_pred_gender})


#    return {'df': pd.concat([df, df_results], axis=1)}