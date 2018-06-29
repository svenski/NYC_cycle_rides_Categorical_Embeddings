
# coding: utf-8

# # Embeddings

# In[13]:

import keras
from keras import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Activation, Flatten

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

import pandas as pd
import numpy as np

from plotnine import *
from plotnine import options
options.set_option('figure_size' , (10,6))

    
import matplotlib.pyplot as plt

import seaborn as sns

from pandas.api.types import CategoricalDtype


# In[14]:

df = pd.read_csv("~/.kaggle/datasets/new-york-city/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")
df['date'] = pd.to_datetime(df['Date'])
df['weekday'] = df['date'].dt.weekday
df['weekday_name'] = df['date'].dt.weekday_name

bridge = 'Total'
by_day = df.groupby('weekday_name')[[bridge]].sum()


# In[16]:

print(ggplot(by_day.reset_index(), aes('weekday_name', bridge)) + geom_bar(stat = 'identity'))

