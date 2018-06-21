import ipyterminit

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
    
import matplotlib.pyplot as plt

import seaborn as sns

from pandas.api.types import CategoricalDtype

def main():
    df = pd.read_csv("~/.kaggle/datasets/new-york-city/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")
    df['date'] = pd.to_datetime(df['Date'])
    df['weekday'] = df['date'].dt.weekday
    df['weekday_name'] = df['date'].dt.weekday_name

    plotWeekdayCounts(df)
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_type = CategoricalDtype(categories=weekdays, ordered=True)
    df['weekday_name'] = df['weekday_name'].astype(weekday_type)
    plotWeekdayCounts(df)
    brigde = 'Brooklyn Bridge'
    plotWeekdayCounts(df,brigde)

    ggplot(df, aes(brigde)) + geom_density(aes(fill = 'weekday_name'), alpha = 0.3) + facet_wrap('~weekday_name')


    df['users'] = df['Brooklyn Bridge']

    df = df[df['users'] > 0]
    df['scaled_users'] = (df['users'] - np.mean(df['users']))/np.std(df['users'])

    emb_size = 3
    embedding_names = [f'D{x+1}' for x in np.arange(emb_size)]

    model = keras.Sequential()
    model.add(Embedding(input_dim=7, output_dim=emb_size, input_length=1, name="embedding"))
    model.add(Flatten())
    model.add(Dense(units=40, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))

    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

    hh = model.fit(x=df[['weekday']], y=df[['scaled_users']], epochs=50,batch_size=2)

    mm = model.get_layer('embedding')
    emb_matrix = mm.get_weights()[0]

    emp_df = pd.DataFrame(emb_matrix, columns = embedding_names)
    emp_df['weekday'] = np.arange(0,7)

    df = pd.merge(df, emp_df, on = 'weekday')

    dummyw = pd.get_dummies(df['weekday_name'])

    df_X = pd.concat([df, dummyw], ignore_index= False, axis = 1)
    y = df_X['Williamsburg Bridge']

    all_x = embedding_names + weekdays
    df_X = df_X[all_x]

    model = LinearRegression()
    cat_x = df_X[weekdays]
    emb_x = df_X[embedding_names]

    bootstrap = ShuffleSplit(n_splits=100,  random_state=0)
    cat_scores = cross_val_score(model, cat_x, y, scoring="neg_mean_squared_error", cv=bootstrap)
    emb_scores = cross_val_score(model, emb_x, y, scoring="neg_mean_squared_error", cv=bootstrap)

    scores = pd.DataFrame({ 'categorical':-cat_scores, 'embedded':-emb_scores})
    scores.mean()

    ggplot(scores, aes('categorical')) + geom_density() + geom_density(aes('embedded'), color = 'red')

    corr_mat = emp_df.drop('weekday', axis = 1).transpose().corr()
    corr_mat.columns = weekdays 
    corr_mat['weekdays_name'] = weekdays
    corr_mat.set_index('weekdays_name', inplace = True)

    plt.matshow(corr_mat)
    plt.colorbar()

    sns.heatmap(corr_mat)
    sns.heatmap(emp_df.drop('weekday', axis = 1))

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(xs = emp_df['D1'], ys = emp_df['D2'], zs = emp_df['D3'], c = emp_df['weekday'])

    for row_num, day in emp_df.iterrows():
        ax.text(x = day['D1'] , y = day['D2'], z = day['D3'], s = weekdays[row_num])
    

def plotWeekdayCounts(df, bridge = 'Total'):
    by_day = df.groupby('weekday_name')[[bridge]].sum()
    print(ggplot(by_day.reset_index(), aes('weekday_name', bridge)) + geom_bar(stat = 'identity'))

