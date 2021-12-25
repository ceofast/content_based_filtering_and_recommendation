# CONTENT BASED FILTEERING AND RECOMMENDATION

# Recommendations are developed based on the similarities of the product ingredients.

# 1. Representing texts mathematically. (Vectorizing Texts)

# 2. Calculating similarity of texts converted to matrix form.
# There are some ways to represent these texts via vectors which methods are;
# Count Vector (Word Count),
# TF-IDF,
# These are the most widely used methods.


# We have a sample dataset about movies.
# In this dataset, we will convert the description part of each movie into a mathematical matrix,
# that is, put it into numbers and find the ones that are similar to each other from these numbers,
# and capture the similarity of the movies to each other.

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_4/the_movies_dataset/movies_metadata.csv", low_memory=False)

df.head()

df.shape

df["overview"].head()

# CountVectrizer Method

corpus = ['This is the first document.',
          'This document is the second document',
          'And this is the third one',
          'Is this the first document?']

#word frequency

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

vectorizer.get_feature_names()

X.toarray()

# TF-IDF Matrix method

# TF-IDF = TF(t) * IDF(t)

# Step 1: TF(t) = (Frequency of a t term observed in the relevant document) / (Total number of terms in the document) (term frequency)
# Step 2: IDF(t) = 1 + log_e((Total number of documents + 1) / (Number of documents with t term + 1) (inverse document frequency)
# Step 3: TF-IDF = TF(t) * IDF(t)
# Step 4: L2 normalization to TF-IDF values

# L2 normalization is on a row basis.
# It is summed by squaring each observation in a row and taking the square root of the total.
# All observations in the row are divided by the values obtained as a result of the square root.

# word tf-idf

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word')

X = vectorizer.fit_transform(corpus)

vectorizer.get_feature_names()

X.toarray()

# Obtaining the TF-IDF for Our Problem

df['overview'].head()

tfidf = TfidfVectorizer(stop_words='english')

df['overview'] = df['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df['overview'])

tfidf_matrix.shape
# (45466, 75827)
# In this output, 45366 lines is the unique movie count.
# In this output, 75827 is the number of columns and indicates the unique word count.

df['title'].shape

# Creating the Cosine Similarity Matrix

# The Cosine Similarity matrix creates a similarity matrix.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape

#We called the similarity score of the 1st movie with other movies.
cosine_sim[1]

# Generating Recommendations Based on Similarities

indices = pd.Series(df.index, index=df['title'])

indices = indices[~indices.index.duplicated(keep='last')]

indices.shape

indices[:10]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

similarity_scores.head(20)

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

movie_indices

df["title"].iloc[movie_indices]

def content_based_recommender(title, cosine_sim, dataframe):
    dataframe = dataframe[~dataframe["title"].isna()]
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    movie_index = indices[title]
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]



content_based_recommender("Sherlock Holmes", cosine_sim, df)
content_based_recommender("The Godfather", cosine_sim, df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)