import pandas as pd
from tqdm import tqdm

movie_file_path = './ml-25m/movies.csv'
movies_df_tmp = pd.read_csv(movie_file_path)

movie_genre = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
               'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
               'Thriller', 'War', 'Western', '(no genres listed)']
columns = ['movieId', 'title'] + movie_genre
movies_df = pd.DataFrame(columns=columns)
movies_df[['movieId', 'title']] = movies_df_tmp[['movieId', 'title']]
for i in tqdm(range(19)):
    for j in range(len(movies_df_tmp)):
        if movie_genre[i] in movies_df_tmp['genres'][j].split('|'):
            movies_df.loc[j, movie_genre[i]] = 1
        else:
            movies_df.loc[j, movie_genre[i]] = 0

print(movies_df.head(5))



# rating_file_path = './ml-25m/ratings.csv'
# ratings_df = pd.read_csv(rating_file_path)
# ratings_df = ratings_df[['userId', 'movieId', 'rating']]
