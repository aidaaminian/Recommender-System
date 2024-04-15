import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
import mlflow


def recommend_top_k(preds_df, ratings_df, movie, userId, k=10):
    user_row = userId - 1
    sorted_user_predictions = preds_df.iloc[user_row].sort_values(ascending=False)
    user_data = ratings_df[ratings_df.userId == (userId)]
    user_rated = user_data.merge(movie, how='left', left_on='movieId', right_on='movieId'). \
        sort_values(['rating'], ascending=False)
    user_preds = movie.merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             on='movieId').rename(columns={user_row: 'prediction'}). \
                     sort_values('prediction', ascending=False). \
                     iloc[:k, :]
    return user_rated, user_preds


def precision_at_k(df, k=10, y_test: str = 'rating', y_pred='prediction'):
    dfK = df.head(k)
    sum_df = dfK[y_pred].sum()
    true_pred = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    if sum_df > 0:
        return true_pred / sum_df
    else:
        return None


def recall_at_k(df, k=10, y_test='rating', y_pred='prediction'):
    dfK = df.head(k)
    sum_df = df[y_test].sum()
    true_pred = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    if sum_df > 0:
        return true_pred / sum_df
    else:
        return None


class CollaborativeModel(mlflow.pyfunc.PythonModel):

    def fit(self, num_components=15, threshold=2):
        """### explore datasets"""

        rating = pd.read_csv('IMDB/ratings_small.csv')
        movie = pd.read_csv('IMDB/movies_metadata.csv')
        movie = movie.rename(columns={'id': 'movieId'})

        """### data preprocessing

        There are three rows entered by mistake, so we remove that row.
        """

        movie = movie[
            (movie['movieId'] != '1997-08-20') & (movie['movieId'] != '2012-09-29') & (
                    movie['movieId'] != '2014-01-01')]

        def find_names(x):
            if x == '':
                return ''
            genre_arr = eval(str(x))
            return ','.join(i['name'] for i in eval(str(x)))

        movie['genres'] = movie['genres'].fillna('')

        movie['genres'] = movie['genres'].apply(find_names)

        movie.movieId = movie.movieId.astype("uint64")

        self.movie = movie

        """only keep rating for movies with metadata in movie dataset"""

        new_rating = pd.merge(rating, movie, how='inner', on=["movieId"])

        new_rating = new_rating[["userId", "movieId", "rating"]]

        self.new_rating = new_rating

        """### matrix factorization"""

        inter_mat_df = rating.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        inter_mat = inter_mat_df.to_numpy()
        ratings_mean = np.mean(inter_mat, axis=1)
        inter_mat_normal = inter_mat - ratings_mean.reshape(-1, 1)

        """We use singular value decomposition for matrix factorization"""

        svd_U, svd_sigma, svd_V = randomized_svd(inter_mat_normal,
                                                 n_components=num_components,
                                                 n_iter=5,
                                                 random_state=47)

        """This function gives the diagonal form"""

        svd_sigma = np.diag(svd_sigma)

        """Making predictions"""

        rating_weights = np.dot(np.dot(svd_U, svd_sigma), svd_V) + ratings_mean.reshape(-1, 1)
        self.weights_df = pd.DataFrame(rating_weights, columns=inter_mat_df.columns)

    def predict(self, context, model_input):
        return self.my_custom_function(model_input)

    def my_custom_function(self, model_input):
        # do something with the model input
        self.fit(15, 2)
        print(model_input)
        self.user_rated, self.user_preds = recommend_top_k(self.weights_df, self.new_rating, self.movie,
                                                           int(model_input), 100)
        return self.user_preds

    def eval_metrics(self):
        df_res = self.user_preds[["movieId", "prediction"]]. \
            merge(self.user_rated[["movieId", "rating"]], how='outer', on='movieId')

        df_res.sort_values(by='prediction', ascending=False, inplace=True)

        df_res['prediction'] = df_res['prediction'] >= threshold
        df_res['rating'] = df_res['rating'] >= threshold
        prec_at_k = precision_at_k(df_res, 100, y_test='rating', y_pred='prediction')
        rec_at_k = recall_at_k(df_res, 100, y_test='rating', y_pred='prediction')

        print("precision@k: ", prec_at_k)
        print("recall@k: ", rec_at_k)
        return prec_at_k, rec_at_k


c_model = CollaborativeModel()
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=c_model)
    num_components = 15
    threshold = 2
    c_model.fit(num_components, threshold)
    c_model.predict(context=pd.DataFrame([]), model_input=220)
    prec_at_k, rec_at_k = c_model.eval_metrics()
    mlflow.log_param("num_components", num_components)
    mlflow.log_param("threshold", threshold)
    mlflow.log_metric("precision_at_k", prec_at_k)
    mlflow.log_metric("recall_at_k", rec_at_k)
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
