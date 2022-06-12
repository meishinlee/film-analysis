import pandas as pd 
import nltk
import numpy as np
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import linear_kernel
from queue import PriorityQueue as pq

import streamlit as st

st.markdown("# Inclusive Movie Recommedations")
st.sidebar.markdown("# Inclusive Movie recommendations")

text = open("datasets/review_ratings.csv")
output = open("datasets/res.txt","w")
# print(text)
text.readline() # remove header
for row in text: 
    row_details = row.split('^')
    title, year, genre, duration = row_details[1].strip(' ,"\''), row_details[2].strip(' ,"\''), row_details[3].strip(' ,"\''), row_details[4].strip(' ,"\'')
    critic_rate, pub_rate = row_details[6].strip(' ,"\''), row_details[7].strip(' ,"\'')
    desc, notes, listed_in, comment = row_details[8].strip(' ,"\''),row_details[9].strip(' ,"\''),row_details[16].strip(' ,"\''),row_details[17].strip(' ,"\'')
    alpha_rate = row_details[16].strip(' ,"\'')

    
    combined_desc = desc+' '+ notes+' '+listed_in+' ' +comment+' ' +title+' '+genre
    formatted_str = title+'\t'+str(year)+('\t')+str(critic_rate)+'\t'+str(pub_rate)+'\t'+combined_desc+'\n'
    output.write(formatted_str)
    # print(desc)

output.close()
text.close()

punc = string.punctuation
films = {}
text = open("datasets/review_ratings.csv")
# print(text)
text.readline() # remove header
for row in text: 
    row_details = row.split('^')
    title, year, genre, duration = row_details[1].strip(' ,"\''), row_details[2].strip(' ,"\''), row_details[3].strip(' ,"\''), row_details[4].strip(' ,"\'')
    critic_rate, pub_rate = row_details[6].strip(' ,"\''), row_details[7].strip(' ,"\'')
    desc, notes, listed_in, comment = row_details[8].strip(' ,"\''),row_details[9].strip(' ,"\''),row_details[16].strip(' ,"\''),row_details[17].strip(' ,"\'')
    alpha_rate = row_details[16].strip(' ,"\'')

    combined_desc = desc+' '+ notes+' '+listed_in+' ' +comment+' ' +title+' '+genre

    valid_words = ""
    for word in combined_desc: 
        if word == " ": 
            valid_words += word
        elif word in punc: 
            pass 
        elif word.isalpha() == False: 
            pass
        else: 
            valid_words += word 
    films[(title, year)] = valid_words
            
    # formatted_str = title+'\t'+str(year)+('\t')+str(critic_rate)+'\t'+str(pub_rate)+'\t'+combined_desc+'\n'

text.close()
# films

st.write("There are",len(films),"films in this dataset")
films_df = pd.DataFrame.from_dict(films,orient="index")
films_df.columns = ['desc']

st.dataframe(films_df.head(5))

# Remove line breaks 
def remove_linebreaks(input):
    text = re.compile(r'\n')
    return text.sub(r' ',input)

films_df["desc"] = films_df["desc"].apply(remove_linebreaks)

# Tokenize words 
nltk.download('punkt')
films_df["desc"] = films_df["desc"].apply(word_tokenize)

# Remove stopwords
nltk.download('stopwords')
def remove_stopwords(input1):
    words = []
    for word in input1:
        if word not in stopwords.words('english'):
            words.append(word)
    return words
films_df["desc"] = films_df["desc"].apply(remove_stopwords)

# Lemmatization
nltk.download('wordnet')
nltk.download('omw-1.4')
lem = WordNetLemmatizer()
def lemma_wordnet(input):
    return [lem.lemmatize(w) for w in input]
films_df["desc"] = films_df["desc"].apply(lemma_wordnet)

def combine_text(input):
    combined = ' '.join(input)
    return combined
films_df["desc"] = films_df["desc"].apply(combine_text)

# Train a TF-IDF vectorizer 
vectorizer = TfidfVectorizer(max_features=1800, lowercase=True, stop_words='english', ngram_range=(1,2)) #unigrams and bigrams

# features 
tf_idf_output = vectorizer.fit_transform(films_df["desc"])
vocab = np.array(vectorizer.get_feature_names())

input_keyword = st.text_input("Enter the keywords relating a movie you want to watch", 'boy romance girl crush love')

st.write("Applying NMF")

num_topics = st.number_input("Pick no of topics", 50)

num_topics = int(num_topics)
nmf = NMF(n_components = num_topics, solver = "mu").fit(tf_idf_output)
W = nmf.fit_transform(tf_idf_output)
H = nmf.components_

num_top_words = st.number_input("Pick no of top words", 20)

num_top_words = int(num_top_words)
def show_topics(a):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]
topics = show_topics(H)
i = 1

topics_set_list = []
for row in topics: 
    topic = row.split()
    #print("Topic",i,":",set(topic))
    topics_set_list.append(set(topic))
    i += 1

st.write("Applying jaccard distance algorithm")

def jaccard_score(topic_set, desc_set):
    intersect = len(topic_set.intersection(desc_set))
    union = len(topic_set.union(desc_set))
    return 1-intersect/union
    # return len(topic_set.intersection(desc_set))

input_keywords_to_set = set(input_keyword.split())
best_topic_score = 0
best_topics = None
for topic in topics_set_list: 
    if jaccard_score(topic, input_keywords_to_set) > best_topic_score: 
        best_topic_score = jaccard_score(topic, input_keywords_to_set) 
        best_topics = topic

st.write("Best similar topics", best_topics)

pq_best_titles = pq()
# max_d = 0
for k, v in films.items(): 
    jd = jaccard_score(best_topics, set(v.split()))
    pq_best_titles.put((jd, k))

num_recs = 0
out_df = pd.DataFrame()
#out_df.columns = ['Title', 'Release Year']
title = []
year = [] 
while pq_best_titles.empty() == False and num_recs < 5:
    #print("Title:",pq_best_titles.get()[1][0],"\tRelease Year:",pq_best_titles.get()[1][1])
    #curr = {'Title': pq_best_titles.get()[1][0], 'Release Year' : pq_best_titles.get()[1][1] }
    title.append(pq_best_titles.get()[1][0])
    year.append(pq_best_titles.get()[1][1])
    num_recs += 1 
out_df['Title'] = title
out_df['Release Year'] = year
st.table(out_df)

    
