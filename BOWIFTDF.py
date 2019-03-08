#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:


get_ipython().run_line_magic('run', 'minute tokenization.py')


# In[2]:


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


# In[ ]:


get_ipython().run_cell_magic('time', '', "os.chdir(r'D:\\lecture\\NLPTA\\project\\code')\ndoc_df = pd.read_csv('doc_df.csv')")


# In[74]:


doc_df.pop(doc_df.columns[0])


# In[ ]:


wnl = WordNetLemmatizer()

corpus = doc_df['content'].apply(lem_term).tolist()
#%% bag of words with sklearn
vectorizer = CountVectorizer(stop_words = 'english',lowercase = True)
AnuualBow = vectorizer.fit_transform(corpus)
df_AnuualBow = pd.DataFrame(AnuualBow.A,columns = vectorizer.get_feature_names())


# In[59]:


for col in ['year','month','day','mr','meeting']:
    try:
        df_AnuualBow.pop(col)
    except:
        continue


# In[64]:


doc_df_BoW_sk2 = pd.concat([doc_df,df_AnuualBow],axis = 1)


# In[66]:


doc_df_BoW_sk2


# In[65]:


'year' in doc_df_BoW_sk2.columns


# In[62]:


##无论是用 LDA还是sklearn，都去不掉mr。
"mr" in df_AnuualBow.columns


# In[ ]:


#%% tf-idf sklearn
v = TfidfVectorizer(stop_words='english', max_df=0.9)
tfidf = v.fit_transform(corpus)
df_Anuualtfidf = pd.DataFrame(tfidf.A,columns = v.get_feature_names())


# In[ ]:


for col in ['year','month','day','mr','meeting']:
    try:
        df_AnuualBow.pop(col)
    except:
        continue


# In[3]:


doc_df_tfidf_sk = pd.concat([doc_df,df_Anuualtfidf],axis = 1)


# In[ ]:


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


# In[29]:


IR = pd.read_csv(r'D:\lecture\NLPTA\project\FEDRateChange.csv')
IR >> head(3)


# In[36]:


IR.columns


# In[63]:


doc_df_BoW_sk2.columns


# In[68]:


doc_df_BoW_sk2.rename(columns={doc_df_BoW_sk2.columns[1]: "Year"}, inplace=True)
doc_df_BoW_sk2.rename(columns={doc_df_BoW_sk2.columns[2]: "Month"}, inplace=True)
doc_df_BoW_sk2.rename(columns={doc_df_BoW_sk2.columns[3]: "Day"}, inplace=True)


# In[39]:


IR['IR_change_freq_Annual'] = IR.groupby(['Year'])['IR_Change'].transform('count')

IR[IR['IR_change_freq_Annual'] == IR['IR_change_freq_Annual'].max()]


# In[37]:


com_col = set(IR.columns&doc_df_BoW_sk2.columns)


# In[72]:


bow_IR = pd.merge(IR,doc_df_BoW_sk2,on = list(com_col),how = 'outer')


# In[73]:


bow_IR >> head(10)


# In[76]:


count_IR_increase = bow_IR.fillna(0)


# In[82]:


count_IR_increase.groupby('Year').sum() >>head(5)


# In[ ]:


tfidf_IR = pd.merge(IR,doc_df_BoW_sk2,on = list(com_col),how = 'outer')


# In[ ]:


doc_df_tfidf_sk.rename(columns={doc_df_tfidf_sk.columns[1]: "Year"}, inplace=True)
doc_df_tfidf_sk.rename(columns={doc_df_tfidf_sk.columns[2]: "Month"}, inplace=True)
doc_df_tfidf_sk.rename(columns={doc_df_tfidf_sk.columns[3]: "Day"}, inplace=True)


# In[ ]:


dfTfidf_IR = pd.merge(IR,doc_df_tfidf_sk,on = list(com_col),how = 'outer')


# In[ ]:


dfTfidf_IR_increase = bow_IR.fillna(0)


# In[ ]:


dfTfidf_IR_increase.groupby('Year').sum() >>head(5)

