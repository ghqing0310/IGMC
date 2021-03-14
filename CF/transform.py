import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse


# 读取数据集 ml-100k
user_df = pd.read_csv(
    '../ml-100k/u.user', names=['user_id', 'age', 'sex', 'occupation', 'zip_code'], header=None, sep='|')

item_df = pd.read_csv(
    '../ml-100k/u.item', names=['movie_id', 'movie title', 'release date', 'video release date', 'IMDb URL',
                                'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
                                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
    header=None, sep='|')

train_df = pd.read_csv(
    '../ml-100k/u1.base', names=['user_id', 'item_id', 'rating', 'timestamp'], header=None, sep='\t')


# print('user_df \n', user_df.head(5))
# print('item_df \n', item_df.head(5))
# print('train_df \n', train_df.head(5))
# print('test_df \n', test_df.head(5))

n_users = user_df.shape[0]
n_items = item_df.shape[0]
print('用户数 %d, 电影数 %d.' % (n_users, n_items))

# 生成训练集和测试集的用户-电影-评分矩阵
train_data = train_df['rating'].tolist()
train_row = (train_df['user_id'].to_numpy()-1).tolist()
train_col = (train_df['item_id'].to_numpy()-1).tolist()
train_m = sparse.coo_matrix(
    (train_data, (train_row, train_col)), shape=(n_users, n_items)).toarray()
train_id_m = (train_m > 0).astype('int')  # 生成指示矩阵，表示用户是否对该电影评分
del train_data, train_row, train_col


# 生成电影-类别的表示矩阵
movie_genre_m = item_df[['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
                         'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']].to_numpy()


# 计算用户-类别矩阵，表示用户对此类别的电影的平均得分，用已有的数据计算
# 注意此时NaN正好表示用户没有对此类别的电影评价过
emb_m = movie_genre_m
user_cluster_m = np.dot(train_m, emb_m) / np.dot(train_id_m, emb_m)

user_index, cluster_index = np.where(user_cluster_m > 0)
rating_list = user_cluster_m[user_index, cluster_index]
rating_list = np.around(rating_list, decimals=6)
user_cluster_df = pd.DataFrame(
    {'user_id': user_index + 1, 'cluster_id': cluster_index + 1, 'rating': rating_list})
print(user_cluster_df.head(5))
# user_cluster_df.to_csv('../useful_data/u.new_rating', index=False, header=False)
