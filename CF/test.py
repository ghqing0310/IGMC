import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# 读取数据
item_df = pd.read_csv('../ml-25m/movies_df.csv').reset_index(drop=False)
rating_df = pd.read_csv(
    '../ml-25m/ratings.csv', names=['user_id', 'item_id', 'rating', 'timestamp'], header=0)
train_df, test_df = train_test_split(rating_df, test_size=0.2, random_state=7)

print('item_df \n', item_df.head(5))
print('train_df \n', train_df.head(5))
print('test_df \n', test_df.head(5))

# 电影的原始id是跳跃的，要重新编号并记录转换的字典
movieId2movie_id = dict(
    zip(item_df['movieId'].tolist(), item_df['index'].tolist()))

n_users = max(rating_df['user_id'])
n_items = len(movieId2movie_id)

# 生成训练集和测试集的用户-电影-评分矩阵
train_data = (train_df['rating']*2).astype('int').tolist()
train_row = (train_df['user_id'].to_numpy()-1).tolist()
train_col = list(
    map(lambda x: movieId2movie_id[x], (train_df['item_id'].to_numpy()).tolist()))
train_m = sparse.coo_matrix(
    (train_data, (train_row, train_col)), shape=(n_users, n_items)).toarray()
train_id_m = (train_m > 0).astype('int')  # 生成指示矩阵，表示用户是否对该电影评分
del train_data, train_row, train_col

test_data = test_df['rating'].tolist()
test_row = (test_df['user_id'].to_numpy()-1).tolist()
test_col = (test_df['item_id'].to_numpy()-1).tolist()
test_m = sparse.coo_matrix(
    (test_data, (test_row, test_col)), shape=(n_users, n_items)).toarray()
del test_data, test_row, test_col

# 生成电影-类别的表示矩阵
movie_genre_m = item_df[['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
                         'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']].to_numpy()
