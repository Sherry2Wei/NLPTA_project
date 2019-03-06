import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dfply import *
import os
from nltk.stem import WordNetLemmatizer
from dfply import *
import os

#%%

from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer

path = r'D:\lecture\NLPTA\project\FOMC minutes\1993'

os.chdir(path)

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
from nltk.tokenize import word_tokenize
