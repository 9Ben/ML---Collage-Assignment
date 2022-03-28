#!/usr/bin/env python
# coding: utf-8

# ## assignment 4 - Nlp ex - Labeling gender from hebrew text. 

# In[1]:


import pandas as pd
import numpy as np
# ------------- visualizations:
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# ---------------------------------------
import sklearn
from sklearn import preprocessing, metrics, pipeline, model_selection, feature_extraction 
from sklearn import naive_bayes, linear_model, svm, neural_network, neighbors, tree
from sklearn import decomposition, cluster

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ----------------- output and visualizations: 
import warnings
from sklearn.exceptions import ConvergenceWarning
# show several prints in one cell. This will allow us to condence every trick in one cell.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# ---------------------------------------


# #### Text analysis and String manipulation imports:
# ###### Stop words are not allowed

# In[2]:


# --------- Text analysis and Hebrew text analysis imports:
# vectorizers:
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# regular expressions:
import re


# In[3]:


train_filename = 'annotated_corpus_for_train.csv'
df_train = pd.read_csv(train_filename, index_col=None, encoding='utf-8')
df_train.head()
df_train.shape


# #### Removing punctuations and duplications

# In[4]:


import string
#removing punctuations 
def remove_punc(text):
    for punc in string.punctuation:
        text = text.replace(punc, '')
    return text
#
df_train['story'] = df_train['story'].apply(remove_punc)
#checking if there are duplicated rows, in case there are any duplicated rows - im only keeping the last one 
df_train = df_train.drop_duplicates(subset=['story'], keep='last')
df_train['story'] = df_train['story'].str.replace('[a-zA-Z0-9]', ' ')
#turning the labels of m/f to 0s & 1s
df_train['label'] = df_train['gender'].map({'m': 0,'f': 1})

df_train.shape


# In[5]:


print(df_train['gender'].value_counts())


# In[6]:


# given a train df which includes labeled corpus with hebrew stories and target values (gender). 
#  test df (not labeled) was also given and i have been asked to predict the gender of the story writter.
# i was also asked not to use any stop words or hebrew tokenizers. 


# Splitting the train Dataframe

# In[7]:


def split_train(df):
    X = df_train['story'].copy()
    y = df_train['label'].copy()
    return X,y


# In[8]:


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42, shuffle=False ) 
    return X_train, X_test, y_train, y_test


# In[9]:


X,y = split_train(df_train)
X_train, X_test, y_train, y_test = split_data(X, y)


# In[10]:


print("X_train shape is:", X_train.shape)
print("y_train shape is:", y_train.shape)
print("X_test shape is:", X_test.shape)
print("y_test shape is:", y_test.shape)

i have been asked to get f1 macro score higher than 74 so i will share few things that i've tried 
# In[11]:


#using a pipeline, built this one with perceptron as my classifier & tfidf - f1 score - 93.
#f1 macro score - 80 

train_pipe = Pipeline([

    ('vectorizer' , TfidfVectorizer(analyzer = 'word' ,ngram_range=(1, 3),max_df=0.8, token_pattern=r'\b\w{2,15}\b')),
    ('clf' , Perceptron(random_state = 42,alpha=0.001, penalty='elasticnet', tol = 1e-7, shuffle = True ,
                        eta0 = 0.101))    
])
train_pipe.fit(X_train,y_train)
y_pred = train_pipe.predict(X_test)
confusion_matrix(y_test,y_pred)
clf_rep = metrics.classification_report(y_test,y_pred)
print(clf_rep)
print(f1_score(y_test, y_pred, average='macro'))


# In[12]:


# tried using sgd classifier as well - f1 score - 92
# f1 macro score - 79

#i will use pipeline to classify 
train_pipe = Pipeline([
     #('vect', CountVectorizer()),
    #ngram_range=(1, 1) analyzer == 'word' \b{4,20}\b  token_pattern=r'(?u)\b\w\w+\b' r'(\w{3,20}\s)\b'
    
    ('vectorizer' , TfidfVectorizer(analyzer = 'word' ,ngram_range=(1, 3),max_df=0.7, token_pattern=r'\b\w{3,10}\b')),
    ('clf' , SGDClassifier(random_state = 42,alpha=0.006, loss='perceptron'))
    
   
    #verbose = 10 # loss='perceptron', 'optimal
])
train_pipe.fit(X_train,y_train)
y_pred = train_pipe.predict(X_test)
confusion_matrix(y_test,y_pred)
clf_rep = metrics.classification_report(y_test,y_pred)
print(clf_rep)
print(f1_score(y_test, y_pred, average='macro'))


# In[ ]:




