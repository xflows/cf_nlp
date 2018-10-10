import pandas as pd


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




        

        







                       


        


    
    
