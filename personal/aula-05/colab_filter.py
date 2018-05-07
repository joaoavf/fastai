import torch

from fastai.learner import *
from fastai.column_data import *


def std_collab_filter(path, val_idxs, n_factors, wd):
    cf = CollabFilterDataset.from_csv(path, 'ratings.csv', 'userId', 'movieId', 'rating')

    learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn=optim.Adam)

    learn.fit(1e-2, 2)
    learn.fit(1e-2, 3, wds=wd, cycle_len=1, cycle_mult=2)


class EmbeddingDotBias(nn.Module):
    def __init__(self, n_users_, n_movies_, n_factors_):
        super().__init__()
        (self.u, self.m, self.ub, self.mb) = [get_emb(*o) for o in [
            (n_users_, n_factors_), (n_movies_, n_factors_), (n_users_, 1), (n_movies_, 1)
        ]]

    def forward(self, cats, conts):

        users, movies = cats[:, 0].unsqueeze(1), cats[:, 1].unsqueeze(1)
        print(users.shape, movies.shape)
        print('ha')
        print(self.u(users).shape)
        print(self.m(movies).shape)
        um = (self.u(users) * self.m(movies)).sum(1)

        print(self.ub(users).squeeze().shape, self.mb(movies).squeeze().shape)
        res = um + self.ub(users).squeeze() + self.mb(movies).squeeze()
        res = F.sigmoid(res) * (5 - 1) + 1
        return res

path = 'ml-latest-small/'

ratings = pd.read_csv(path + 'ratings.csv')
movies = pd.read_csv(path + 'movies.csv')

val_idxs = get_cv_idxs(len(ratings))
wd = 2e-4
n_factors = 50

# User 2 Index
u_uniq = ratings.userId.unique()
user2idx = {o: i for i, o in enumerate(u_uniq)}
ratings.userId = ratings.userId.apply(lambda x: user2idx[x])

# Movie 2 Index
m_uniq = ratings.movieId.unique()
movie2idx = {o: i for i, o in enumerate(m_uniq)}
ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])

n_users = int(ratings.userId.nunique())
n_movies = int(ratings.movieId.nunique())

x = ratings.drop(['rating', 'timestamp'], axis=1)
y = ratings['rating']

data = ColumnarModelData.from_data_frame(path, val_idxs, x, y, ['userId', 'movieId'], 64)

wd = 1e-5
model = EmbeddingDotBias(n_users, n_movies, 50)
opt = optim.SGD(model.parameters(), 1e-1, weight_decay=wd, momentum=0.9)

set_lrs(opt, 0.01)

fit(model, data, 3, opt, F.mse_loss)