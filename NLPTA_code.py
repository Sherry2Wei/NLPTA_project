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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import statsmodels.api as sm
minutes = pd.read_csv(r'D:\lecture\NLPTA\project\code local back up\doc_df.csv')

# 1. data processing
minutes.pop(minutes.columns[0]s)
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

stop_words = stopwords.words('english')

import datetime
Month = [datetime.date(2008, i, 1).strftime('%B').lower() for i in range(1,13)]
stop_words.extend(['year','month','day','mr','meeting','committee','ms','federal','page']
                    + Month)

import gensim
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

texts = list(sent_to_words(minutes['content']))

bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)

from gensim.utils import simple_preprocess
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in data:
        doc = nlp(re.sub('\_',''," ".join(sent)))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

corpus_no_stops = remove_stopwords(texts)
corpus_bigrams = make_bigrams(corpus_no_stops)
set(['infla_tion' in word for word in corpus_bigrams])
data_lemmatized = lemmatization(corpus_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

corpus = [' '.join(wordList) for wordList in data_lemmatized]
set(['infla_tion' in word for word in corpus])

# bag of words with sklearn
vectorizer = CountVectorizer(stop_words = 'english',lowercase = True)
AnnualBow = vectorizer.fit_transform(corpus)
df_AnnualBow = pd.DataFrame(AnnualBow.A,columns = vectorizer.get_feature_names())
for term in ['year','month','day']:
    try:
        df_AnnualBow.pop(term)
    except:
        continue

'month' in df_AnnualBow.columns
##remove thoes meaningness words
minutes_BoW_sk2 = pd.concat([minutes,df_AnnualBow],axis = 1)

# tf-idf sklearn
v = TfidfVectorizer(stop_words='english', max_df=0.9)
tfidf = v.fit_transform(corpus)
df_Annualtfidf = pd.DataFrame(tfidf.A,columns = v.get_feature_names())

for term in ['year','month','day']:
    try:
        df_Annualtfidf.pop(term)
    except:
        continue

minutes_tfidf_sk = pd.concat([minutes,df_Annualtfidf],axis = 1)

# import interest rate data and merge them
IR = pd.read_csv(r'D:\lecture\NLPTA\project\FEDRateChange.csv')
IR >> head(3)

minutes_BoW_sk2.rename(columns={minutes_BoW_sk2.columns[0]: "Year"}, inplace=True)
minutes_BoW_sk2.rename(columns={minutes_BoW_sk2.columns[1]: "Month"}, inplace=True)
minutes_BoW_sk2.rename(columns={minutes_BoW_sk2.columns[2]: "Day"}, inplace=True)

com_col = set(IR.columns&minutes_BoW_sk2.columns)
bow_IR = pd.merge(IR,minutes_BoW_sk2,on = list(com_col),how = 'outer')
bow_IR.sort_values(by = ['Year','Month','Day'],inplace = True)
bow_IR.fillna(0,inplace = True)
bow_IR >> head(5)

minutes.head(10)
minutes_tfidf_sk.rename(columns={minutes_tfidf_sk.columns[0]: "Year"}, inplace=True)
minutes_tfidf_sk.rename(columns={minutes_tfidf_sk.columns[1]: "Month"}, inplace=True)
minutes_tfidf_sk.rename(columns={minutes_tfidf_sk.columns[2]: "Day"}, inplace=True)

tfIdf_IR = pd.merge(IR,minutes_tfidf_sk,on = list(com_col),how = 'outer')
tfIdf_IR.sort_values(by = ['Year','Month','Day'],inplace =True)
# data analysis -- correlation

def CorTerms(terms,df_sum,y,top = None,bottom = None):
    correlations = [np.correlate(y,df_sum[term])[0]
                    for term in list(terms)]## change the index
    IR_corTerms = pd.DataFrame({'keyterms':terms,'correlations':correlations})
    top = IR_corTerms.sort_values(by = 'correlations',ascending = False) >> head(top)
    bottom = IR_corTerms.sort_values(by = 'correlations',ascending = True) >> head(bottom)

    return IR_corTerms,top,bottom

def corBar(x,y):
    plt.barh(range(len(x)), y, height=0.7, color='steelblue', alpha=0.8) 
    plt.yticks(range(len(x)), x)
    plt.xlabel("correlations")
    plt.ylabel('keyterms')
    plt.title(" correlations with IR change")
    plt.show()

"""
## First I performed analysis on BoW, it seems like tfIdf is better. 
## you may just skip this
keyterms = list(bow_IR.columns[8:])
cor_cols = list(bow_IR.columns[0:6]) + keyterms
bow_IR_cor = bow_IR[cor_cols]
bow_IR_cor2 = bow_IR_cor.groupby(['Year','Month']).sum()
bow_IR_cor[['IR_Change']] >> head(10)
#
CorBowSum = CorTerms(keyterms,bow_IR_cor2,bow_IR_cor2['IR_Change'],top = 10,bottom = 10)
BoW_IR_SumTop = CorBowSum[1]
BoW_IR_SumBottom = CorBowSum[2]
# plot graph
## first with positive correlations
corBar(BoW_IR_SumTop['keyterms'],BoW_IR_SumTop['correlations'])
corBar(BoW_IR_SumBottom['keyterms'],BoW_IR_SumBottom['correlations'])

## count IR_change
bow_IR >> head(3)
bow_IR.columns
bow_IR.fillna(0,inplace = True)
keyterms = bow_IR.columns[8:]
bow_IR_sum = bow_IR.groupby('Year').agg('sum')
IR_change_freq = bow_IR.groupby('Year')['IR_Change'].agg('count')
##bow_IR_freq.insert(0,'IR_change_freq',IR_change_freq)

bow_IR.head(3).groupby('Year').apply(np.sum)
## the correlation between keyterms and IR change freq

CorBowFreq = CorTerms(keyterms,bow_IR_sum,IR_change_freq,top = 10,bottom = 10)

top20_freq = CorBowFreq[1]
bottom20_freq = CorBowFreq[2]
corBar(top20_freq['keyterms'],top20_freq['correlations'])
corBar(bottom20_freq['keyterms'],bottom20_freq['correlations']) ## this is nonsense

## at first attempt, I regressed the frequency with the bow, this time, The R square was improved
## eventhough the coef is not that significant
"""
# the problem with bow is that we assign equal weight to all terms. Therefore, some words that 
# every documents was given high coef in our regression model, which creat too much noise. 
# Next,we may try the above method on tf-idf model.
tfIdf_IR.fillna(0,inplace = True)
tfIdf_IR[['Year','IR_Change']].head(20)
tfIdf_IR.groupby('Year')['IR_Change'].sum() >> head(50)
tfIdf_sum = tfIdf_IR.groupby('Year').agg('sum')
len(tfIdf_sum)

tfIdf_sum_sub = tfIdf_sum.query('Year > 1990')
CorTfidfIR = CorTerms(tfIdf_sum_sub.columns[5:],tfIdf_sum_sub,tfIdf_sum_sub['IR_Change'],top = 20,bottom = 20)

tfIdf_top = CorTfidfIR[1]
tfIdf_bottom = CorTfidfIR[2]
corBar(tfIdf_top['keyterms'],tfIdf_top['correlations'])
corBar(tfIdf_bottom['keyterms'],tfIdf_bottom['correlations'])

## logistic regression
### according to the output above, tfidf is better. So 
### I will continue with tfidf

### 1. classify IR increase or decrease
IR_ChID= np.where(tfIdf_sum_sub['IR_Change']>0,1,0)
tfIdf_sum_sub.insert(4,'IR_ChID',IR_ChID)
tfIdf_sum_sub.shape
## to improve the model, I would filter out thoes insignifiant terms
from sklearn.feature_selection import f_regression
words = tfIdf_sum_sub.columns[6:]
X = tfIdf_sum_sub[words] 
y = tfIdf_sum_sub['IR_ChID']

Fvalue = f_regression(X, y, center=True)[0]
Pvalue = f_regression(X, y, center=True)[1]

stat_CorTfidfIR = CorTfidfIR[0]
stat_CorTfidfIR['Fvalue'] = Fvalue
stat_CorTfidfIR['Pvalue'] = Pvalue

signTerms = stat_CorTfidfIR.query('Pvalue < 0.05')

signTerms['Cor_P'] = signTerms['correlations'] /signTerms['Pvalue']*signTerms['Fvalue']
signTermsBottom = signTerms.sort_values(by='correlations', ascending=True) >> head(20)
signTermsTop = signTerms.sort_values(by = 'correlations', ascending=False) >> head(20)

corBar(signTermsTop['keyterms'],signTermsTop['correlations'])
corBar(signTermsBottom['keyterms'],signTermsBottom['correlations'])

ID_var = signTerms['keyterms'].tolist()
X_sign = tfIdf_sum_sub[ID_var]
y_sign = IR_ChID

from sklearn.model_selection import train_test_split
X_train_sign,X_test_sign,y_train_sign,y_test_sign=train_test_split(X_sign,y_sign,test_size=0.15,random_state=0)

## it seems that our model workd quite well and also we checked the R-square

## From here I am gonna compare different classifiacation models.
## And find the best model to predict

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": MultinomialNB(),
    #"AdaBoost": AdaBoostClassifier(),
    #"QDA": QuadraticDiscriminantAnalysis(),
    #"Gaussian Process": GaussianProcessClassifier()
}

def batch_classify(X_train, Y_train, X_test, Y_test, no_classifiers = 5, verbose = True):
    
    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return dict_models
 
 
def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    return(df_.sort_values(by=sort_by, ascending=False))


dict_models = batch_classify(X_train_sign, y_train_sign, X_test_sign, y_test_sign, no_classifiers = 8)
result = display_dict_models(dict_models)
result.to_csv('regressionCompare.csv')

## as the result shows, Logistic regression and Naive Bayes turned out to 
## be the best classifiers.
### 1.run logistic regression
logreg = LogisticRegression(fit_intercept= True)
words = tfIdf_sum_sub.columns[6:]
model_logistic = logreg.fit(X_sign, y_sign)

model_logistic.score(X_sign,y_sign)
y_pred_sign=logreg.predict(X_test_sign)

cnf_matrix_sign = metrics.confusion_matrix(y_test_sign, y_pred_sign)
cnf_matrix_sign

import seaborn as sns
## cnf_matrix visualiztion. 
class_names=['success','fail'] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix_sign), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
## or we could try to see what the result would be if we don't 
## exclude the insinificant words.

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0)
logreg.fit(X,y)
y_pred=logreg.predict(X_test)

cnf_matrix_logistic = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix_logistic

model_logistic2 = logreg.fit(X,y)
model_logistic2.score(X,y)

cnf_matrix2 = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix2

### 2.naive bayes regression

from sklearn.naive_bayes import MultinomialNB
NaiveMulti = MultinomialNB()
modelNaiveMulti = NaiveMulti.fit(X_sign, y_sign)
modelNaiveMulti.score(X_sign,y_sign)

y_pred_Navive=NaiveMulti.predict(X_test_sign)

cnf_matrix3 = metrics.confusion_matrix(y_test_sign, y_pred_Navive)
cnf_matrix3

## According to the cnf_matrix, logistic regression is better.

## with insignificant terms
modelNaiveMulti2 = NaiveMulti.fit(X, y)
modelNaiveMulti2.score(X,y)

y_pred_Navive2=NaiveMulti.predict(X_test)
cnf_matrix4 = metrics.confusion_matrix(y_test, y_pred_Navive2)
cnf_matrix4

# try unsupervised learning 
## what are the main topics in the minutes? 
## or, say what's the main concern when Fed decide interest rate?



