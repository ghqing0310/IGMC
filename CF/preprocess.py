import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(flag):
    if flag not in ['ml-100k', 'ml-1m']:
        print('Input Error!')
        return

    if flag == 'ml-100k':
        # 读取数据集 ml-100k
        user_df = pd.read_csv(
            './ml-100k/u.user', names=['user_id', 'age', 'sex', 'occupation', 'zip_code'], header=None, sep='|')

        item_df = pd.read_csv(
            './ml-100k/u.item', names=['movie_id', 'movie title', 'release date', 'video release date', 'IMDb URL',
                                        'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
                                        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                                        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
            header=None, sep='|')

        train_df = pd.read_csv(
            './ml-100k/ua.base', names=['user_id', 'item_id', 'rating', 'timestamp'], header=None, sep='\t')

        test_df = pd.read_csv(
            './ml-100k/ua.test', names=['user_id', 'item_id', 'rating', 'timestamp'], header=None, sep='\t')
    else:
        # 读取数据集 ml-1m
        user_df = pd.read_csv('./ml-1m/users.dat', names=[
            'user_id', 'age', 'sex', 'occupation', 'zip_code'], header=None, sep='::', engine='python')

        # 注意电影id不是顺序连续的，要重编码一下
        item_df = pd.read_csv(
            './ml-1m/movies.csv', names=['movie_id', 'movie title', 'unknown', 'Action', 'Adventure', 'Animation',
                                          'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                          'Thriller', 'War', 'Western'], header=0)
        item_df = item_df.reset_index(drop=False)
        item_df['index'] = item_df['index'] + 1
        map_df = item_df[['movie_id', 'index']]

        # rating中的编号也要变换一下
        rating_df = pd.read_csv('./ml-1m/ratings.dat', names=[
                                'user_id', 'item_id', 'rating', 'timestamp'], header=None, sep='::', engine='python')
        rating_df = rating_df.merge(
            map_df, how='left', left_on='item_id', right_on='movie_id')
        rating_df = rating_df.drop(columns=['item_id', 'movie_id']).rename(
            columns={'index': 'item_id'})
        train_df, test_df = train_test_split(
            rating_df, test_size=0.2, random_state=7)
    return user_df, item_df, train_df, test_df
