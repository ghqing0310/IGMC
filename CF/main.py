import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from preprocess import read_data
import warnings


np.seterr(divide='ignore', invalid='ignore')


def main(flag):
    # 读取数据，有'ml-100k'和'ml-1m'两种
    user_df, item_df, train_df, test_df = read_data(flag)
    print('user_df \n', user_df.head(5))
    print('item_df \n', item_df.head(5))
    print('train_df \n', train_df.head(5))
    print('test_df \n', test_df.head(5))

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

    test_data = test_df['rating'].tolist()
    test_row = (test_df['user_id'].to_numpy()-1).tolist()
    test_col = (test_df['item_id'].to_numpy()-1).tolist()
    test_m = sparse.coo_matrix(
        (test_data, (test_row, test_col)), shape=(n_users, n_items)).toarray()
    del test_data, test_row, test_col

    # 生成电影-类别的表示矩阵
    movie_genre_m = item_df[['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
                             'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']].to_numpy()

    # 对电影做聚类，并把label做one-hot编码
    # cluster = KMeans(n_clusters=19)
    # cluster.fit_predict(movie_genre_m)
    # labels = cluster.labels_
    # ohe_m = OneHotEncoder(sparse=False).fit_transform(labels.reshape((-1, 1)))
    # emb_m = ohe_m

    # 计算用户-类别矩阵，表示用户对此类别的电影的平均得分，用已有的数据计算
    # 注意此时NaN正好表示用户没有对此类别的电影评价过
    emb_m = movie_genre_m
    user_cluster_m = np.dot(train_m, emb_m) / np.dot(train_id_m, emb_m)

    # 用nanmean和nansum处理非空数据
    row_mean = np.nanmean(user_cluster_m, axis=1)
    row_norm = np.nansum(user_cluster_m ** 2, axis=1) ** 0.5

    # 对非空数据做标准化，然后将空值替换成0
    standard_m_nan = user_cluster_m - row_mean.reshape(-1, 1)
    normalized_m = standard_m_nan / row_norm.reshape(-1, 1)
    normalized_m[np.isnan(normalized_m)] = 0

    # 生成用户之间的相似度矩阵 Pearson Correlation Similarity
    per_cor_sim_m = np.dot(normalized_m, normalized_m.T)

    # 找到每个用户最相似的top_k用户的下标
    top_k = 150 if flag == 'ml-100k' else 300
    top_k_id_m = per_cor_sim_m.argsort()[:, -top_k:][:, ::-1]

    # 对所有的nan做值预测
    nan_id = np.where(np.isnan(standard_m_nan))
    for i, j in zip(nan_id[0], nan_id[1]):
        # 第i个用户对第j个类别的评分，用最相似的用户的第j个类别的平均分做预测值
        warnings.simplefilter("ignore")
        v = np.nanmean(standard_m_nan[top_k_id_m[i]][:, j])
        if not np.isnan(v):
            standard_m_nan[i, j] = v
        else:
            standard_m_nan[i, j] = 0
        # print('第%d行 第%d列的值已预测完毕！' % (i, j), end="", flush=True)

    result = standard_m_nan + row_mean.reshape(-1, 1)
    result[result < 1] = 1
    result[result > 5] = 5
    np.save('user_cluster_' + flag[3:] + '.npy', result)
    np.save('users_' + flag[3:] + '.npy', per_cor_sim_m)

    # 测试
    test_id = np.where(test_m > 0)
    y_true = []
    y_pred = []
    for i, j in zip(test_id[0], test_id[1]):
        y_true.append(test_m[i, j])
        y_pred.append(result[i].dot(movie_genre_m[j]) / sum(movie_genre_m[j]))
        # y_pred.append(result[i, labels[j]])
    print('Root Mean Squared Error: % f' %
          np.sqrt(mean_squared_error(y_true, y_pred)))


if __name__ == '__main__':
    flag = 'ml-100k'
    main(flag)
