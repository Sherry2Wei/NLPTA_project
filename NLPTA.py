import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dfply import *
import re
from nltk.stem import WordNetLemmatizer
from dfply import *
import os
from nltk.tokenize import word_tokenize
#%%
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
path = r'D:\lecture\NLPTA\project\FOMC minutes\1993'
os.chdir(path)
#%%
file_names = os.listdir(path)
file_type = ['txt' in file_name for file_name in file_names]
df_file = pd.DataFrame({'name':file_names,'type':file_type},)
files_93 = df_file.query('type == True')
content = [open(file).read() for file in files_93['name']]
def data_clean(data):##clean special characters
    return re.sub("\d+|\-|\n|/|:", "", data)
content = [data_clean(open(file).read()) for file in files_93['name']]
files_93['content'] = content
files_93
files93 = files_93.reset_index()
files93.pop('index')

files93['day'] = files93['name'].apply(lambda x : re.findall('\d+',x)[0][-2:])
files93['month'] = files93['name'].apply(lambda x : re.findall('\d+',x)[0][-4:-2])
files93['year'] = files93['name'].apply(lambda x : re.findall('\d+',x)[0][:-4])
files93 = files93[['year','month','day','name','type','content']]

files93
wnl = WordNetLemmatizer()

def lem_term(x):
    tokens = word_tokenize(x)
    lem_token= [wnl.lemmatize(word) for word in tokens]
    for char in (['.',',']):
        while char in lem_token:
            lem_token.remove(char)
    content_lem = ' '.join(lem_token)
    return content_lem
corpus = files93['content'].apply(lem_term).tolist()
vectorizer = CountVectorizer(stop_words = 'english',lowercase = True)
AnuualBow = vectorizer.fit_transform(corpus)
df_AnuualBow = pd.DataFrame(AnuualBow.A,columns = vectorizer.get_feature_names())
files93_BoW_sk2 = pd.concat([files93,df_AnuualBow],axis = 1)
v = TfidfVectorizer(stop_words='english', max_df=0.9)
tfidf = v.fit_transform(corpus)
df_Anuualtfidf = pd.DataFrame(tfidf.A,columns = v.get_feature_names())
files93_tfidf_sk = pd.concat([files93,df_Anuualtfidf],axis = 1)
files93_tfidf_sk

## nltk bag of words
#%%
def get_bow_nltk(content):
    tokens = [w for w in word_tokenize(content.lower()) if w.isalpha()]
    no_stops = [t for t in tokens if t not in stopwords.words('english')]
    bow_nltk = Counter(no_stops).most_common(len(no_stops))
    df_bow_nltk = pd.DataFrame(bow_nltk,columns = ['terms','occurrences'])
    df_bow_nltk['content'] = content*len(no_stops)
    return df_bow_nltk

#%%
bow_nltk_combine = pd.concat([get_bow_nltk(content) for content in corpus])

#%% gensim tf-idf
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

tokenized_contents = [word_tokenize(content.lower()) for content in corpus]
#%% we need to remove the special characters
d2 = Dictionary(tokenized_contents)
bow_corpus = [d2.doc2bow(doc) for doc in tokenized_contents if doc.isapha()]
tfidf_gensim = TfidfModel(bow_corpus)

tfidf_weights2 = tfidf_gensim[bow_corpus[0]]