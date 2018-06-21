import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.ion()

from IPython import get_ipython
get_ipython().magic("load_ext autoreload")
get_ipython().magic("%autoreload 2")
import pandas as pd
import numpy as np

###################
import keras
from keras import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Activation, Flatten

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

df = pd.read_csv("/home/paperspace/.kaggle/datasets/new-york-city/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")
df['date'] = pd.to_datetime(df['Date'])
df['weekday'] = df['date'].dt.weekday
df['weekday_name'] = df['date'].dt.weekday_name
df['users'] = df['Brooklyn Bridge']

df = df[df['users'] > 0]
df['scaled_users'] = (df['users'] - np.mean(df['users']))/np.std(df['users'])

emb_size = 4

model = keras.Sequential()
model.add(Embedding(input_dim=7+1, output_dim=emb_size, input_length=1, name="embedding"))
model.add(Flatten())
model.add(Dense(units=40, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

hh = model.fit(x=df[['weekday']], y=df[['scaled_users']], epochs=50,batch_size=2)

mm = model.get_layer('embedding')
emb_matrix = mm.get_weights()[0]

emp_df = pd.DataFrame(emb_matrix, columns = ['D1','D2','D3', 'D4'])
emp_df['weekday'] = np.arange(0,8)



df = pd.merge(df, emp_df, on = 'weekday')

dummyw = pd.get_dummies(df['weekday_name'])

df_X = pd.concat([df, dummyw], ignore_index= False, axis = 1)
y = df_X['Williamsburg Bridge']
emb_names = ['D1', 'D2', 'D3', 'D4']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekdays_unknown = weekdays + ['unknown']

# emp_df['weekday_name'] = weekdays_unknown
# emp_df.set_index('weekday_name')

all_x = emb_names + weekdays
df_X = df_X[all_x]

#X_train, X_test, y_train, y_test = train_test_split(df_X, y)
model = LinearRegression()
cat_x = df_X[weekdays]
emb_x = df_X[emb_names]

from sklearn.model_selection import ShuffleSplit

bootstrap = ShuffleSplit(n_splits=100,  random_state=0)
cat_scores = cross_val_score(model, cat_x, y, scoring="neg_mean_squared_error", cv=bootstrap)
emb_scores = cross_val_score(model, emb_x, y, scoring="neg_mean_squared_error", cv=bootstrap)

scores = pd.DataFrame({ 'categorical':-cat_scores, 'embedded':-emb_scores})
scores.mean()

from plotnine import *

ggplot(scores, aes('categorical')) + geom_density() + geom_density(aes('embedded'), color = 'red')

import matplotlib.pyplot as plt
plt.matshow(emp_df.drop('weekday', axis = 1).transpose().corr())

by_day = df.groupby('weekday_name')[['Total']].sum()
ggplot(by_day.reset_index(), aes('weekday_name', 'Total')) + geom_bar(stat = 'identity')
