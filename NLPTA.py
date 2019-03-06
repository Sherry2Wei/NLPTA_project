#!/usr/bin/env python
# coding: utf-8

# In[3]:



import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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


# ### <font face = 'Times New Roman'> Lemmatize

# In[380]:


wnl = WordNetLemmatizer()

def lem_term(x):
    tokens = word_tokenize(x)
    lem_token= [wnl.lemmatize(word) for word in tokens]
    for char in (['.',',']):
        while char in lem_token:
            lem_token.remove(char)
    content_lem = ' '.join(lem_token)
    return content_lem


# In[385]:


files93


# In[383]:


get_ipython().run_cell_magic('time', '', "corpus = files93['content'].apply(lem_term).tolist()")


# In[384]:


corpus


# ### sklearn

# ### <font face = 'Times New Roman'> Get BoW with sklearn

# In[169]:


vectorizer = CountVectorizer(stop_words = 'english',lowercase = True)


# In[316]:


AnuualBow = vectorizer.fit_transform(corpus)


# In[317]:


df_AnuualBow = pd.DataFrame(AnuualBow.A,columns = vectorizer.get_feature_names())


# In[318]:


files93_BoW_sk2 = pd.concat([files93,df_AnuualBow],axis = 1)


# ### <font face = 'Times New Roman'> Get tf-df with sklearn

# In[178]:


v = TfidfVectorizer(stop_words='english', max_df=0.9)


# In[321]:


tfidf = v.fit_transform(corpus)


# In[322]:


df_Anuualtfidf = pd.DataFrame(tfidf.A,columns = v.get_feature_names())


# In[323]:


files93_tfidf_sk = pd.concat([files93,df_Anuualtfidf],axis = 1)


# In[324]:


files93_tfidf_sk


# ### <font face = 'Times New Roman'> Get BoW with nltk

# In[24]:


from nltk.tokenize import word_tokenize


# In[32]:


from nltk.corpus import stopwords


# In[172]:


tokens = [w for w in word_tokenize(contents[0].lower()) if w.isalpha()] ## isalpha 判断是否为英文字符


# In[173]:


no_stops = [t for t in tokens if t not in stopwords.words('english')]


# In[174]:


np.array(Counter(no_stops).most_common(10))


# In[54]:


df = pd.DataFrame(np.array(Counter(no_stops).most_common(10)),columns = ['words','occurency'])


# In[55]:


df


# In[17]:


from collections import Counter


# In[46]:


Counter(tokens).most_common(2)


# ### lemmatize
# 
# 

# In[270]:


lem_term(contents[0])


# In[264]:


files['']


# In[263]:


tokens = word_tokenize(contents[0])


# In[265]:


lem_token= [wnl.lemmatize(word) for word in tokens]


# In[275]:


lem_token.remove(',')


# In[281]:





# In[282]:


lem_token


# In[273]:


re.sub('.|/,','',lem_token)


# In[253]:


lem_term(files93['content'][0])


# In[251]:


[lem_term(files93['content'][i]) for i in range(len(files93))]


# In[248]:


files93['content'].transform(lambda x:lem_term(x))


# In[254]:


word_tokenize(contents[0])


# In[236]:


bow = vectorizer.fit_transform(lem_token)


# In[ ]:





# In[237]:


len(AnuualBow.A[0])


# In[238]:


len(bow.A[0])


# In[201]:


len(tokens)


# In[204]:


x = vectorizer.fit_transform([contents[0]])


# In[209]:


len(vectorizer.get_feature_names())


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




