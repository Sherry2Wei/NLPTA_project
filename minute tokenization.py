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
    path = r"c:\Users\ChonWai\Desktop\NPL\Data\FOMCminutes"
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
#%%

    import spacy
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    # NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    import gensim
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

#%%
import pandas as pd

minutes = doc_df
corpus = list(sent_to_words(minutes['content']))

corpus[0]

# Build the bigram and trigram models
bigram = gensim.models.Phrases(corpus, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[corpus], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

import spacy

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

corpus_no_stops = remove_stopwords(corpus)

corpus_bigrams = make_bigrams(corpus_no_stops)

nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

data_lemmatized = lemmatization(corpus_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:1])
#%%

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

print(lda_model.print_topics())
doc_lda = lda_model[corpus]
print('\nPerplexity: ', lda_model.log_perplexity(corpus))

coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

pyLDAvis.display(vis)









