import os
from dfply import *
import time
import pandas as pd
import re
#from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool


def readfile(filename):
    file = open(path + os.sep + filename, encoding="utf8")
    file_contents = file.read()
    return file_contents


def lem_term(document):
    wnl = WordNetLemmatizer()
    document = re.sub('\\_|\\+d', '', document)
    tokens = word_tokenize(document)
    lem_token = [wnl.lemmatize(word) for word in tokens]
    for char in (['.', ',']):
        while char in lem_token:
            lem_token.remove(char)
    document = ' '.join(lem_token)
    return document


if __name__ == '__main__':
    start_time = time.time()
    path = r"D:\lecture\NLPTA\project\FOMCminutes"
    doc_list = os.listdir(path)
    doc_content_list = list()
    [doc_content_list.append(readfile(doc)) for doc in doc_list]
    # Create a document dataframe
    doc_name = [i[:-4] for i in doc_list]
    doc_df = pd.DataFrame({"file_name": doc_name})
    doc_df['day'] = doc_df['file_name'].apply(lambda x: re.findall('\\d+', x)[0][-2:])
    doc_df['month'] = doc_df['file_name'].apply(lambda x: re.findall('\\d+', x)[0][-4:-2])
    doc_df['year'] = doc_df['file_name'].apply(lambda x: re.findall('\\d+', x)[0][:-4])
    doc_df = doc_df[['year', 'month', 'day', 'file_name']]
    # Tokenize each document in document_list
    p = Pool(processes=7)
    clean_doc_content_list = list(p.map(lem_term, doc_content_list))
    doc_df["content"] = clean_doc_content_list
    print(doc_df >> head(5))
    print("--- %s seconds ---" % (time.time() - start_time))










