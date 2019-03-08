
# import file through ipython
## in cmd:>> activate [conda enviroment]
         #>> ipython
%cd D:\lecture\NLPTA\project\code local back up\
%run minute_tokenization.py

minutes = doc_df
import os
import pandas as pd
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
#minutes = doc_df
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['meeting', 'committee', 'january', 'february', 'mr'])
import gensim
import spacy
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


texts = list(sent_to_words(minutes['content']))

bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[texts], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


corpus_no_stops = remove_stopwords(texts)
corpus_bigrams = make_bigrams(corpus_no_stops)

%%time
data_lemmatized = lemmatization(corpus_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
words_list = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in words_list]

# View
print(corpus[:1])
%%time
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10,
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
#%% run below codes in ipython with magic command
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

%matplotlib inline

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

#%% find dominant topics
for i,row in enumerate(lda_model[corpus]):
    print(i,row)
  sent_topics_df = pd.DataFrame()
  row = sorted(row,key = lambda x:x[1],reverse =True)
  for j, (topic_num, prop_topic) in enumerate(row):
    if j == 0:  # => dominant topic
      wp = lda_model.print_topic(topic_num)
      topic_keywords = ", ".join([word for word, prop in wp])
      sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), 	  round(prop_topic,4), topic_keywords]), ignore_index=True)
    else:   
      break

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=corpus):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):## i refers to documnet number
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Per Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row): ##each row consists of topic number and prop_topic
            if j == 0:  # => dominant topic
                wp = lda_model.print_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=corpus)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)