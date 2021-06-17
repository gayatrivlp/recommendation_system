#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd


# In[12]:


'''understanding the data'''


# In[3]:


credits = pd.read_csv("G:/RUTGERS/512/final_project/movielens/credits.csv")


# In[4]:


keywords = pd.read_csv("G:/RUTGERS/512/final_project/movielens/keywords.csv")


# In[5]:


links_small= pd.read_csv("G:/RUTGERS/512/final_project/movielens/links_small.csv")


# In[6]:


md= pd.read_csv("G:/RUTGERS/512/final_project/movielens/movies_metadata.csv",low_memory=False)


# In[8]:


ratings = pd.read_csv("G:/RUTGERS/512/final_project/movielens/ratings_small.csv")


# In[8]:


md.iloc[0:3].transpose()


# In[9]:


links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')


# In[24]:


'''preprocessing'''


# In[10]:


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


# In[11]:


#converting the id column of md into int

md['id'] = md['id'].apply(convert_int)


# In[12]:


#cleaning the data
#dropping the movies with no id
md[md['id'].isnull()]
md = md.drop([19730, 29503, 35587])


# In[13]:


md['id'] = md['id'].astype('int')


# In[14]:


#forming a small movie dataset whose ids match with those in links_small
smd = md[md['id'].isin(links_small)]


# In[15]:


smd.head()


# In[16]:


'''we build a recommensdation system using movie description and taglines'''


# In[17]:


#replacing all NaN taglines in small movies data
smd['tagline'] = smd['tagline'].fillna('')


# In[19]:


smd['description'] = smd['overview'] + smd['tagline']


# In[20]:


smd['description'] = smd['description'].fillna('')


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# In[24]:


tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])


# In[25]:


tfidf_matrix.shape


# In[27]:


from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


# In[28]:


# http://scikit-learn.org/stable/modules/metrics.html#linear-kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[29]:



cosine_sim[0]


# In[30]:


smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])


# In[31]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[32]:


get_recommendations('The Godfather').head(10)


# In[33]:


get_recommendations('The Dark Knight').head(10)


# In[34]:


'''collaborative filtering'''


# In[2]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


# In[3]:


reader = Reader()


# In[10]:


data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#data.split(n_folds=5)


# In[12]:


svd = SVD()


# In[14]:


cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[15]:


trainset = data.build_full_trainset()
svd.train(trainset)


# In[16]:


ratings[ratings['userId'] == 1]


# In[17]:


svd.predict(1, 302)

