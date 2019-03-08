#!/usr/bin/env python
# coding: utf-8



import os
from dfply import *
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool
import warnings
from nltk.stem import WordNetLemmatizer
warnings.filterwarnings("ignore",category=DeprecationWarning)


minutes = pd.read_csv(r'D:\lecture\NLPTA\project\code local back up\doc_df.csv')


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

minutes.pop(minutes.columns[0])

wnl = WordNetLemmatizer()

corpus = minutes['content'].apply(lem_term).tolist()

#%% bag of words with sklearn
vectorizer = CountVectorizer(stop_words = 'english',lowercase = True)
AnnualBow = vectorizer.fit_transform(corpus)
df_AnnualBow = pd.DataFrame(AnnualBow.A,columns = vectorizer.get_feature_names())


##remove thoes meaningness words
for col in ['year','month','day','mr','meeting','committee']:
    try:
        df_AnnualBow.pop(col)
    except:
        continue
##check:
'year' in df_AnnualBow.columns

minutes_BoW_sk2 = pd.concat([minutes,df_AnnualBow],axis = 1)

##无论是用 LDA还是sklearn，都去不掉mr。
"mr" in df_AnnualBow.columns

#%% tf-idf sklearn
v = TfidfVectorizer(stop_words='english', max_df=0.9)
tfidf = v.fit_transform(corpus)
df_Annualtfidf = pd.DataFrame(tfidf.A,columns = v.get_feature_names())

for col in ['year','month','day','mr','meeting']:
    try:
        df_Annualtfidf.pop(col)
    except:
        continue

minutes_tfidf_sk = pd.concat([minutes,df_Annualtfidf],axis = 1)

#%% bag of word with nltk
"""""
from collections import Counter
from nltk.corpus import stopwords
def get_bow_nltk(content):
    tokens = [w for w in word_tokenize(content.lower()) if w.isalpha()]
    no_stops = [t for t in tokens if t not in stopwords.words('english')]
    bow_nltk = Counter(no_stops).most_common(len(no_stops))
    df_bow_nltk = pd.DataFrame(bow_nltk,columns = ['terms','occurrences'])
    df_bow_nltk['content'] = content*len(no_stops)
    return df_bow_nltk
"""""


#%% import interest rate data and merge them
IR = pd.read_csv(r'D:\lecture\NLPTA\project\FEDRateChange.csv')
IR >> head(3)

minutes_BoW_sk2.rename(columns={minutes_BoW_sk2.columns[0]: "Year"}, inplace=True)
minutes_BoW_sk2.rename(columns={minutes_BoW_sk2.columns[1]: "Month"}, inplace=True)
minutes_BoW_sk2.rename(columns={minutes_BoW_sk2.columns[2]: "Day"}, inplace=True)


#IR['IR_change_freq_Annual'] = IR.groupby(['Year'])['IR_Change'].transform('count')

##IR[IR['IR_change_freq_Annual'] == IR['IR_change_freq_Annual'].max()]


com_col = set(IR.columns&minutes_BoW_sk2.columns)
bow_IR = pd.merge(IR,minutes_BoW_sk2,on = list(com_col),how = 'outer')
bow_IR.fillna(0,inplace = True)
bow_IR >> head(5)


#count_IR_increase = bow_IR.fillna(0)

#count_IR_increase.groupby('Year').sum() >>head(5)

#%% merge with tfidf
tfidf_IR = pd.merge(IR,minutes_BoW_sk2,on = list(com_col),how = 'outer')

minutes_tfidf_sk.rename(columns={minutes_tfidf_sk.columns[1]: "Year"}, inplace=True)
minutes_tfidf_sk.rename(columns={minutes_tfidf_sk.columns[2]: "Month"}, inplace=True)
minutes_tfidf_sk.rename(columns={minutes_tfidf_sk.columns[3]: "Day"}, inplace=True)


dfTfidf_IR = pd.merge(IR,minutes_tfidf_sk,on = list(com_col),how = 'outer')

#dfTfidf_IR_increase = bow_IR.fillna(0)
#dfTfidf_IR_increase.groupby('Year').sum() >>head(5)

#%% data analysis -- correlation

for i,col in enumerate(bow_IR.columns):
    print(i,col)
keyterms = list(bow_IR.columns[8:])
cor_cols = list(bow_IR.columns[0:6]) + keyterms
bow_IR_cor = bow_IR[cor_cols]
bow_IR_cor2 = bow_IR_cor.groupby(['Year','Month']).sum()
#%%
correlations = [np.correlate(bow_IR_cor2['IR_Change'],bow_IR_cor2[term])[0]
                for term in list(bow_IR.columns[8:])]

IR_corKeyterm = pd.DataFrame({'keyterms':keyterms,'correlations':correlations})
bottom10 = IR_corKeyterm.sort_values(by = 'correlations') >> head(20)
top10 = IR_corKeyterm.sort_values(by = 'correlations') >> tail(20)
#%% plot graph
## first with positive correlations
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

correlation = top10['correlations']
keyterms = top10['keyterms']
plt.barh(range(len(keyterms)), correlation, height=0.7, color='steelblue', alpha=0.8) 
plt.yticks(range(len(keyterms)), keyterms)
plt.xlabel("correlations")
plt.ylabel('keyterms')
plt.title("positive correlations with IR change")
plt.show()


## second with negative correlations

plt.barh(range(len( bottom10['keyterms'])),  bottom10['correlations'], height=0.7, color='steelblue', alpha=0.8)
plt.yticks(range(len( bottom10['keyterms'])), keyterms)
plt.xlabel("correlations")
plt.ylabel('keyterms')
plt.title("negative correlations with IR change")
plt.show()


#%% run regression with human intervention
import rpy2
print(rpy2.__version__)
import statsmodels.api as sm
terms_select =top10['keyterms'].tolist() + bottom10['keyterms'].tolist() + ['IR_Change']
IR_regKeyterm = bow_IR_cor2[terms_select]
x = IR_regKeyterm[terms_select[1:]]
x = sm.add_constant(x)
y = IR_regKeyterm['IR_Change']
model = sm.OLS(y,x).fit()
model.summary() ## the results seems overfit

from sklearn.linear_model import LinearRegression

IR_regKeyterm =pd.get_dummies(IR_regKeyterm)
linear_reg_bow = LinearRegression()

fig, axes = plt.subplots(1,len(IR_regKeyterm.columns.values),sharey=True,constrained_layout=True,figsize=(30,15))

for i,e in enumerate(IR_regKeyterm.columns):
  linear_reg_bow.fit(IR_regKeyterm[e].values[:,np.newaxis], y.values)
  axes[i].set_title("Best fit line")
  axes[i].set_xlabel(str(e))
  axes[i].set_ylabel('SalePrice')
  axes[i].scatter(IR_regKeyterm[e].values[:,np.newaxis], y,color='g')
  axes[i].plot(IR_regKeyterm[e].values[:,np.newaxis],
  linear_reg_bow.predict(IR_regKeyterm[e].values[:,np.newaxis]),color='k')