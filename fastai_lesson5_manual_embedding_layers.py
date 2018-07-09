from fastai.learner import *
from fastai.column_data import *

from kaggleUtils.kaggleUtils import printAllPandasColumns

def main():

    printAllPandasColumns()
    path='~/data/ml-latest-small/'

    ratings = pd.read_csv(path+'ratings.csv')
    movies = pd.read_csv(path+'movies.csv')
    
    gg = movies[['movieId','genres']]
    genres_full = pd.merge(ratings[['movieId']], gg, on = 'movieId', how = 'left')
    genres_full_one_hot = genres_full.drop('movieId', axis =1)['genres'].str.get_dummies(sep="|")
    genre_dummies = pd.concat([movies[['movieId']], genres_full_one_hot], axis=1)

    val_idxs = get_cv_idxs(len(ratings))

    u_uniq = ratings.userId.unique()
    user2idx = {o:i for i,o in enumerate(u_uniq)}
    ratings.userId = ratings.userId.apply(lambda x: user2idx[x])

    m_uniq = ratings.movieId.unique()
    movie2idx = {o:i for i,o in enumerate(m_uniq)}
    idx2movie = {i:o for i,o in enumerate(m_uniq)}
    ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])

    n_users=int(ratings.userId.nunique())
    n_movies=int(ratings.movieId.nunique())

    x = ratings.drop(['rating', 'timestamp'],axis=1)
    # x1 = pd.merge(x, genre_dummies, on='movieId')
    y = ratings['rating'].astype(np.float32)

    min_rating,max_rating = ratings.rating.min(),ratings.rating.max()

    model, user_emb_matrix, movie_emb_matrix = userMovieEmbedding(x, y, val_idxs, min_rating, max_rating)
    # genre_model, rest = genreEmbedding(genre_dummies, y, val_idxs, min_rating, max_rating)
    model, user_emb_matrix, movie_emb_matrix = genreEmbedding(x, genre_dummies, y, val_idxs, min_rating, max_rating)



# TOOD: T-SNE
# TODO: add genre
# TODO: add genre normalisation
# TODO: Use genre as uniqe combination
# TODO: 

def genreEmbedding(x, genre_dummies, y, val_idxs, min_rating, max_rating):

    x_genre_dummies = genre_dummies.drop('movieId', axis=1).astype("uint8")
    n_genres = x_genre_dummies.shape[1]

    # use cats for genre
    cats = x_genre_dummies.columns.values

    x_user_dummies = pd.get_dummies(x['userId'])
    x_movies_dummies = pd.get_dummies(x['movieId'])

    dummy_x = pd.concat([x_user_dummies, x_movies_dummies, x_genre_dummies],  axis=1)
    dummy_x['actual_dummy']=1

    data = ColumnarModelData.from_data_frame('', val_idxs, dummy_x, y, ['actual_dummy'], 64)

    n_factors = 50
    n_factors_genre = 10

    class ManualEmbeddingNetGenre(nn.Module):
        def __init__(self, n_users, n_movies, nh=10, p1=0.05, p2=0.5):
            super().__init__()
            self.users = nn.Linear(n_users, n_factors)
            self.movies = nn.Linear(n_movies, n_factors)
            self.genres = nn.Linear(n_genres, n_factors_genre)

            self.lin1 = nn.Linear(n_factors*2 + n_factors_genre, nh)
            self.lin2 = nn.Linear(nh, 1)
            self.drop1 = nn.Dropout(p1)
            self.drop2 = nn.Dropout(p2)
            
        def forward(self, cats, conts):
            users = conts[:, :n_users]
            movies = conts[:, n_users:(n_users+n_movies)] 
            genres = conts[:, n_genres:] 

            user_emb = self.users(users)
            movies_emb = self.movies(movies)
            genres_emb = self.genres(genres)
            x = self.drop1(torch.cat([user_emb, movies_emb, genres_emb], dim=1))
            x = self.drop2(F.relu(self.lin1(x)))
            return F.sigmoid(self.lin2(x)) * (max_rating-min_rating+1) + min_rating-0.5

    wd=1e-5
    model = ManualEmbeddingNetGenre(n_users, n_movies).cuda()
    opt = optim.Adam(model.parameters(), 1e-3, weight_decay=wd)

    fit(model, data, 3, opt, F.mse_loss)
    set_lrs(opt, 1e-2)
    fit(model, data, 3, opt, F.mse_loss)
    set_lrs(opt, 1e-3)
    fit(model, data, 3, opt, F.mse_loss)

    user_emb_matrix = np.transpose(model.users.weight.data.cpu().numpy())
    movie_emb_matrix = np.transpose(model.movies.weight.data.cpu().numpy())

    return model, user_emb_matrix, movie_emb_matrix


def userMovieEmbedding(x, y, val_idxs, min_rating, max_rating):
    x_user_dummies = pd.get_dummies(x['userId'])
    x_movies_dummies = pd.get_dummies(x['movieId'])

    dummy_x = pd.concat([x_user_dummies, x_movies_dummies], axis=1)
    # It seems like the columnar data set expect at least one categorical column. This will be ingnored later
    dummy_x['actual_dummy']=1

    data = ColumnarModelData.from_data_frame('', val_idxs, dummy_x, y, ['actual_dummy'], 64)

    n_factors = 50
    class ManualEmbeddingNet(nn.Module):
        def __init__(self, n_users, n_movies, nh=10, p1=0.05, p2=0.5):
            super().__init__()
            self.users = nn.Linear(n_users, n_factors)
            self.movies = nn.Linear(n_movies, n_factors)

            self.lin1 = nn.Linear(n_factors*2, nh)
            self.lin2 = nn.Linear(nh, 1)
            self.drop1 = nn.Dropout(p1)
            self.drop2 = nn.Dropout(p2)
            
        def forward(self, cats, conts):
            users = conts[:, :n_users]
            movies = conts[:, n_users:] 

            user_emb = self.users(users)
            movies_emb = self.movies(movies)
            x = self.drop1(torch.cat([user_emb, movies_emb], dim=1))
            x = self.drop2(F.relu(self.lin1(x)))
            return F.sigmoid(self.lin2(x)) * (max_rating-min_rating+1) + min_rating-0.5

    wd=1e-5
    model = ManualEmbeddingNet(n_users, n_movies).cuda()
    opt = optim.Adam(model.parameters(), 1e-3, weight_decay=wd)

    fit(model, data, 3, opt, F.mse_loss)
    set_lrs(opt, 1e-2)
    fit(model, data, 3, opt, F.mse_loss)
    set_lrs(opt, 1e-3)
    fit(model, data, 3, opt, F.mse_loss)

    user_emb_matrix = np.transpose(model.users.weight.data.cpu().numpy())
    movie_emb_matrix = np.transpose(model.movies.weight.data.cpu().numpy())

    return model, user_emb_matrix, movie_emb_matrix


def viz():
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 3)
    movie_pca = pca.fit(movie_emb_matrix.T).components_

    movie_names = movies.set_index('movieId')['title'].to_dict()
    g=ratings.groupby('movieId')['rating'].count()
    topMovies=g.sort_values(ascending=False).index.values[:3000]
    #topMovieIdx = np.array([cf.item2idx[o] for o in topMovies])

    fac0 = movie_pca[0]
    movie_comp = [(f, movie_names[idx2movie[i]]) for f,i in zip(fac0, np.arange(len(fac0)))]
    mm = [movie_comp[i] for i in topMovies]

    sorted(mm, key=itemgetter(0))[:15]
    sorted(mm, key=itemgetter(0), reverse=True)[:15]

    from sklearn.manifold import TSNE
