import os
from dfply import *
import time
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool
import warnings
from nltk.stem import WordNetLemmatizer
warnings.filterwarnings("ignore",category=DeprecationWarning)


def lem_term(document):
    document = re.sub('\d+|\_', '', document)
    wnl = WordNetLemmatizer()
    tokens = word_tokenize(document)
    lem_token = [wnl.lemmatize(word) for word in tokens]
    for char in (['.', ',']):
        while char in lem_token:
            lem_token.remove(char)
    document = ' '.join(lem_token)
    re.sub('\+d','',document)
    return document
os.chdir(r'D:\lecture\NLPTA\project\code')
doc_df.to_csv('doc_df.csv')
wnl = WordNetLemmatizer()

corpus = doc_df['content'].apply(lem_term).tolist()
#%% bag of words with sklearn
vectorizer = CountVectorizer(stop_words = 'english',lowercase = True)
AnuualBow = vectorizer.fit_transform(corpus)
df_AnuualBow = pd.DataFrame(AnuualBow.A,columns = vectorizer.get_feature_names())
doc_df_BoW_sk2 = pd.concat([doc_df,df_AnuualBow],axis = 1)
#%% tf-idf sklearn
v = TfidfVectorizer(stop_words='english', max_df=0.9)
tfidf = v.fit_transform(corpus)
df_Anuualtfidf = pd.DataFrame(tfidf.A,columns = v.get_feature_names())
doc_df_tfidf_sk = pd.concat([doc_df,df_Anuualtfidf],axis = 1)
doc_df_tfidf_sk


#%% bag of word with nltk
from collections import Counter
from nltk.corpus import stopwords
def get_bow_nltk(content):
    tokens = [w for w in word_tokenize(content.lower()) if w.isalpha()]
    no_stops = [t for t in tokens if t not in stopwords.words('english')]
    bow_nltk = Counter(no_stops).most_common(len(no_stops))
    df_bow_nltk = pd.DataFrame(bow_nltk,columns = ['terms','occurrences'])
    df_bow_nltk['content'] = content*len(no_stops)
    return df_bow_nltk

IR = pd.read_csv(r'D:\lecture\NLPTA\project\FEDRateChange.csv')
IR >> head(3)
IR.columns
IR.rename(columns={IR.columns[2]: "day"}, inplace=True)
IR['IR_change_freq_Annual'] = IR.groupby(['Year'])['IR_Change'].transform('count')

IR[IR['IR_change_freq_Annual'] == IR['IR_change_freq_Annual'].max()]

#%% merge interest rate with terms frame
com_col = set(IR.columns&doc_df_BoW_sk2.columns)
#pd.merge(IR,df_AnuualBow,on = list(com_col),how = 'outer')
#len(df_AnuualBow)
#Counter(doc_df_BoW_sk2.columns.tolist())
#len(doc_df_BoW_sk2.columns)

#%% spacy