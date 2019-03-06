#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dfply import *
import os
#%%


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer


# In[12]:


from nltk.stem import WordNetLemmatizer


# In[3]:


path = r'D:\lecture\NLPTA\project\FOMC minutes\1993'


# In[4]:


os.chdir(path)


# In[5]:


file_names = os.listdir(path)

file_type = ['txt' in file_name for file_name in file_names]

df_file = pd.DataFrame({'name':file_names,'type':file_type},)

files_93 = df_file.query('type == True')

content = [open(file).read() for file in files_93['name']]

def data_clean(data):##clean special characters
    return re.sub("\d+|\-|\n|/|:", "", data)

content = [data_clean(open(file).read()) for file in files_93['name']]


# In[6]:


files_93['content'] = content


# In[7]:


files_93

files93 = files_93.reset_index()
files93.pop('index')


# In[8]:


files93


# ### bag of words

# In[ ]:





# In[9]:


vectorizer = CountVectorizer(stop_words = 'english',lowercase = True)


# In[10]:


X = vectorizer.fit_transform([files93['content'][0]])


# In[13]:


tokens = vectorizer.get_feature_names()


# ### lemmatize
# 
# 

# In[15]:


wnl = WordNetLemmatizer()


# In[19]:


len(tokens)


# In[18]:


len([wnl.lemmatize(word) for word in tokens])


# In[17]:


files93


# In[ ]:


df  = pd.DataFrame([re.findall('\d+',file_name) for file_name in file_names],columns = ['A','B','C','D'])


# In[ ]:


df['Doc_type'] = [re.sub('\d+','',file_name).split('.')[0] for file_name in file_names]


# In[ ]:


df['minutes'] = ['minutes' in term for term in df['Doc_type']]


# In[ ]:


df.query('minutes == True')


# In[ ]:


file_contents = file.read()
print(file_contents)


# In[ ]:


from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize, TweetTokenizer


# In[ ]:


sens = sent_tokenize(file_contents)


# In[ ]:


sens[0]


# In[ ]:


sen2= sens[0].replace('-','').replace('\n','').replace('\r','')


# In[ ]:


regexp_tokenize('This is a great #NLP exercise.', r'\w+')


# In[ ]:


token_didgit = regexp_tokenize(s,r'\w+') ##s


# In[ ]:


set(token_didgit)


# In[ ]:


s = re.sub("\d+|\-|\n|/|:", "", sens[0])


# In[ ]:


s


# In[ ]:


type(file_contents)


# In[ ]:


vectorizer = CountVectorizer(stop_words = 'english',lowercase = True)


# In[ ]:


X = vectorizer.fit_transform([s])


# In[ ]:


columns = vectorizer.get_feature_names()


# In[ ]:


columns


# In[ ]:


pd.DataFrame(X.toarray(),columns = columns)


# In[ ]:


v = TfidfVectorizer(stop_words='english', max_df=0.9)


# In[ ]:


tfidf = v.fit_transform(s)


# In[ ]:


v.get_feature_names()     


# In[ ]:


tfidf.toarray()


# In[ ]:


df_tfidf = pd.DataFrame(tfidf.toarray(),columns = v.get_feature_names())


# In[ ]:


df_tfidf


# In[ ]:


x = np.arange(10)


# In[ ]:


condlist = [x<3, x>5]


# In[ ]:


choicelist = [x, x**2]


# In[ ]:


np.select(condlist, choicelist)


# In[ ]:


condlist


# In[ ]:


choicelist


# In[ ]:




