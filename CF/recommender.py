import pandas as pd
from tqdm import tqdm
import numpy as np

movies_df_tmp = pd.read_csv('./test.csv', header=None, names=['movieId', 'title', 'genres'])


movie_genre = ['(no genres listed)', 'Action', 'Adventure', 'Animation',
               'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama',
               'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
               'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
columns = ['movieId', 'title'] + movie_genre
movies_df = pd.DataFrame(columns=columns)
movies_df[['movieId', 'title']] = movies_df_tmp[['movieId', 'title']]
for i in range(19):
    for j in range(len(movies_df_tmp)):
        if movie_genre[i] in movies_df_tmp['genres'][j].split('|'):
            movies_df.loc[j, movie_genre[i]] = 1
        else:
            movies_df.loc[j, movie_genre[i]] = 0
movie_names = movies_df['title'].tolist()

movies_df = movies_df.drop(['movieId', 'title'], axis=1)
# movies_df.to_csv('./movies_new_df.csv')

movie_np = movies_df.values
movie_np = np.dot(np.diag(1 / sum(movie_np.T)), movie_np)

weight_matrix = np.load('user_cluster_100k.npy')

new_ratings = np.dot(weight_matrix, movie_np.T)


def get_top_n_new(i, top_n): # noqa
    movies_id = np.argsort(-new_ratings[i][:])[:top_n].tolist()
    print('The Top ' + str(top_n) + ' new movies we recommend to user ' + str(i) + ' are:' + '\n')
    for j in range(top_n):
        print(str(j+1) + ':    ' + movie_names[movies_id[j]] + '\n')


user_index = int(input("Please input the index of user:" + '\n'))
top_n = int(input("Please input the number of movies you want to recommend to him/her?" + '\n'))
bool_a = int(input("Please input 1 or 2, 1 for existing movies, 2 for new movies:" + '\n'))
if user_index < 1 or user_index > 943:
    print("Index out of range!!!")
elif bool_a == 1:
    pass
elif bool_a == 2:
    get_top_n_new(user_index, top_n)
