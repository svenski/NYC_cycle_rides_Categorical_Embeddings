from torch import nn
from torch import optim

import torch.nn.functional as F

from fastai.column_data import ColumnarModelData
from fastai.learner import fit
from fastai.learner import set_lrs
from fastai.learner import get_cv_idxs

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score

from keras import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Activation, Flatten

import pandas as pd
import numpy as np

from plotnine import *
    
import matplotlib.pyplot as plt

import seaborn as sns

from pandas.api.types import CategoricalDtype

from mpl_toolkits.mplot3d import Axes3D

from collections import namedtuple

from kaggleUtils.kaggleUtils import printAllPandasColumns

def calculateEmbeddingScoreUsing(embeddingMatrixFunction, df, emb_size, embedding_names, batch_size, cross_val_model):
    emb_mat = embeddingMatrixFunction(emb_size, embedding_names, df, batch_size=batch_size)
    X = pd.merge(df[['weekday']], emb_mat, on = 'weekday').drop('weekday', axis=1)
    cv_score = cross_val_score(cross_val_model.model, X, normalise(cross_val_model.y), cv=cross_val_model.bootstrap, scoring='r2')
    return(emb_mat, X, cv_score)


def main():
    printAllPandasColumns()
    df = pd.read_csv("~/.kaggle/datasets/new-york-city/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")

    df = cleanAndTransform(df)

    df['users'] = df['Brooklyn Bridge']
    df = df[df['users'] > 0]
    df['scaled_users'] = normalise(df['users'])

    emb_size = 3
    batch_size = 2
    embedding_names = [f'D{x+1}' for x in np.arange(emb_size)]

    CrossValidationModel = namedtuple('CrossValidationModel', ['model', 'y', 'bootstrap'])
    linearCrossValidation = CrossValidationModel(LinearRegression(), df['Williamsburg Bridge'], ShuffleSplit(n_splits=100))

    EmbeddingData = namedtuple('Embedding', ['embedding_matrix', 'X', 'cv_score'])

    torch_emb = EmbeddingData(*calculateEmbeddingScoreUsing(calculateTorchEmbeddingMatrix, df, emb_size, embedding_names, batch_size, linearCrossValidation))
    torch_man_emg = EmbeddingData(* calculateEmbeddingScoreUsing(calculateTorchManualEmbeddingMatrix, df, emb_size, embedding_names, batch_size, linearCrossValidation))
    keras_emb = EmbeddingData(*calculateEmbeddingScoreUsing(calculateKerasEmbeddingMatrix, df, emb_size, embedding_names, batch_size, linearCrossValidation))

    dummy_X = pd.get_dummies(df['weekday_name'])
    dummy_scores = cross_val_score(model, dummy_X, y, cv=bootstrap, scoring='r2')
    dummy_emb = EmbeddingData(None, dummy_X, dummy_scores)


    scores = pd.DataFrame({'torch': torch_emb.scores, 
        'keras':keras_emb.scores, 
        'dummy':dummy_scores,
        'manual_torch':torch_man_emg.scores,
        'ind' : np.arange(len(dummy_scores))})

    scores_df = scores.set_index('ind').stack().reset_index()
    scores_df.columns = ['ind','type','r2']

    ggplot(scores_df, aes('type','r2')) + geom_boxplot()


def crossValUsingStatsmodel():
    import statsmodels.api as sm
    
    keras_rsq = []
    torch_rsq = []
    dummy_rsq = []
    
    y = normalise(df['Williamsburg Bridge'])

    num_simulations = 100

    for i in np.arange(num_simulations):
        val_ids = get_cv_idxs(len(df), seed = np.random.randint(0,10000,1))

        keras_rsq.append(rsq_for(val_ids, keras_X, y))
        torch_rsq.append(rsq_for(val_ids, torch_X, y))
        dummy_rsq.append(rsq_for(val_ids, dummy_X, y))

    scores = pd.DataFrame({'torch':torch_rsq, 'keras':keras_rsq, 'dummy':dummy_rsq, 'ind' : np.arange(num_simulations)})
    scores_df = scores.set_index('ind').stack().reset_index()
    scores_df.columns = ['ind','type','r2']

    ggplot(scores_df, aes('type','r2')) + geom_boxplot()

    
    
def normalise(x):
    return (x-np.mean(x))/np.std(x)

def cleanAndTransform(df):
    df['date'] = pd.to_datetime(df['Date'])
    df['weekday'] = df['date'].dt.weekday
    df['weekday_name'] = df['date'].dt.weekday_name

    # make the weekdays ordered categories 
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_type = CategoricalDtype(categories=weekdays, ordered=True)
    df['weekday_name'] = df['weekday_name'].astype(weekday_type)
    return(df)


def correlationMatrixFor(emp_df):
    corr_mat = emp_df.drop('weekday', axis = 1).transpose().corr()
    corr_mat.columns = weekdays 
    corr_mat['weekdays_name'] = weekdays
    corr_mat.set_index('weekdays_name', inplace = True)
    return(corr_mat)
    
def seabornHeatmap(corr_mat):
    sns.heatmap(corr_mat)
    sns.heatmap(emp_df.drop('weekday', axis = 1))


def chartEmbedding3D(emp_df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(xs = emp_df['D1'], ys = emp_df['D2'], zs = emp_df['D3'], c = emp_df['weekday'])

    for row_num, day in emp_df.iterrows():
        ax.text(x = day['D1'] , y = day['D2'], z = day['D3'], s = weekdays[row_num])


def calculateKerasEmbeddingMatrix(emb_size, embedding_names, df, batch_size=2):
    model = Sequential()
    model.add(Embedding(input_dim=7, output_dim=emb_size, input_length=1, name="embedding"))
    model.add(Flatten())
    model.add(Dense(units=40, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))

    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

    hh = model.fit(x=df[['weekday']], y=df[['scaled_users']], epochs=50,batch_size=batch_size)

    mm = model.get_layer('embedding')
    emb_matrix = mm.get_weights()[0]

    emp_df = pd.DataFrame(emb_matrix, columns = embedding_names)
    emp_df['weekday'] = np.arange(0,7)

    return(emp_df)

def calculateTorchEmbeddingMatrix(emb_size, embedding_names, df, batch_size):
    n_days = 7

    val_idx = get_cv_idxs(len(df))
    data = ColumnarModelData.from_data_frame('', val_idx, df[['weekday']], df['scaled_users'], ['weekday'], 2)
    
    def get_emb(num_cat, num_emb):
        e = nn.Embedding(num_cat, num_emb)
        e.weight.data.uniform_(-0.01,0.01)
        return(e)

    class weekdayEmbedding(nn.Module):
        def __init__(self, n_days):
            super().__init__()
            self.weekdays = get_emb(n_days, emb_size)
            self.lin1 = nn.Linear(emb_size, 40)
            self.lin2 = nn.Linear(40, 10)
            self.lin3 = nn.Linear(10,1)
            #self.drop1 = nn.Dropout(0.5)
        
        def forward(self, cats, conts):
            weekdays = cats[:,0]
            x = self.weekdays(weekdays)
            x = F.relu((self.lin1(x)))
            x = F.relu((self.lin2(x)))
            return(self.lin3(x))

    model = weekdayEmbedding(n_days).cuda()
    opt = optim.Adam(model.parameters(), 1e-3)
    fit(model, data, 30, opt, F.mse_loss)
    fit(model, data, 30, opt, F.mse_loss)

    emb_matrix =  model.weekdays.weight.data.cpu().numpy()

    emp_df = pd.DataFrame(emb_matrix, columns = embedding_names)
    emp_df['weekday'] = np.arange(0,7)
    # list(model.parameters())

    return(emp_df)


def calculateTorchManualEmbeddingMatrix(emb_size, embedding_names, df, batch_size):
    n_days = 7

    val_idx = get_cv_idxs(len(df))
    dummy_X = pd.get_dummies(df['weekday_name'])
    cols = dummy_X.columns.values.astype(str)
    
    data = ColumnarModelData.from_data_frame('', val_idx, dummy_X, df['scaled_users'], [], 2)

    class weekdayEmbeddingManual(nn.Module):
        def __init__(self, n_days):
            super().__init__()
            self.emb = nn.Linear(n_days, emb_size)
            self.lin1 = nn.Linear(emb_size, 40)
            self.lin2 = nn.Linear(40, 10)
            self.lin3 = nn.Linear(10,1)
            #self.drop1 = nn.Dropout(0.5)
        
        def forward(self, cats, conts):

            x = self.emb(conts)
            x = F.relu((self.lin1(x)))
            x = F.relu((self.lin2(x)))
            return(self.lin3(x))

    model = weekdayEmbeddingManual(n_days).cuda()
    opt = optim.Adam(model.parameters(), 1e-3)
    fit(model, data, 30, opt, F.mse_loss)
    fit(model, data, 30, opt, F.mse_loss)

    emb_matrix = np.transpose(model.emb.weight.data.cpu().numpy())

    emp_df = pd.DataFrame(emb_matrix, columns = embedding_names)
    emp_df['weekday'] = np.arange(0,7)
    # list(model.parameters())

    return(emp_df)


def plotWeekdayCounts(df, bridge = 'Total'):
    by_day = df.groupby('weekday_name')[[bridge]].sum()
    print(ggplot(by_day.reset_index(), aes('weekday_name', bridge)) + geom_bar(stat = 'identity'))


def exploration(df):
    brigde = 'Brooklyn Bridge'
    plotWeekdayCounts(df, brigde)

    brigde = 'Williamsburg Bridge'
    plotWeekdayCounts(df,brigde)
    ggplot(df, aes(brigde)) + geom_density(aes(fill = 'weekday_name'), alpha = 0.3) + facet_wrap('~weekday_name')

def rsq_for(val_ids, keras_X, y):
    y_train, y_val = split_by(val_ids, y)
    keras_train, keras_val = split_by(val_ids, keras_X)

    keras_model = sm.OLS(y_train, keras_train).fit()
    y_pred = keras_model.predict(keras_val)
    return(r2_score(y_val, y_pred))

def split_by(validation_idx, df_raw):
    if(isinstance(df_raw, pd.DataFrame) or isinstance(df_raw, pd.Series)):
        raw_valid = df_raw.iloc[validation_idx]
        raw_train = df_raw.loc[~df_raw.index.isin(raw_valid.index)]
    else:
        raw_valid = np.take(df_raw, validation_idx)
        raw_train = np.delete(df_raw, validation_idx)

    return raw_train, raw_valid
