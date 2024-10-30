"""Functions for cleaning the extracted data"""
import nltk
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")


class InvalidDataCleaner(BaseEstimator, TransformerMixin):
    """Cleans invalid values from the dataset"""

    def __init__(self, popularity_lower=0, popularity_upper=100):
        """Instantiates a custom data cleaner"""
        self.popularity_lower = popularity_lower
        self.popularity_upper = popularity_upper
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        """Fits the data"""
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transforms the data"""
        X = X.dropna()
        print(f"After dropna: {X.shape[0]} rows")

        X = self._feature_selection(X)
        print(f"After feature selection: {X.shape[0]} rows")

        X = self._clean_duplicate_track_ids(X)
        print(f"After removing duplicates: {X.shape[0]} rows")

        X = self._clean_bounded_column(
            X, 'popularity', self.popularity_lower, self.popularity_upper)

        for col in ['duration_ms', 'tempo']:
            X = self._clean_bounded_column(X, col, lower=0, upper=None)

        for col in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']:
            X = self._clean_bounded_column(X, col, lower=0, upper=1)

        X = X[X['key'].isin(range(-1, 12))]
        print(f"After filtering key: {X.shape[0]} rows")

        X = X[X['time_signature'].isin(range(3, 8))]
        print(f"After filtering time_signature: {X.shape[0]} rows")

        X = X[X['mode'].isin([0, 1])]
        print(f"After filtering mode: {X.shape[0]} rows")

        X['track_name_sentiment'] = self._get_track_name_sentiment(
            X['track_name'])

        return X

    def _clean_bounded_column(self, data, col_name, lower=None, upper=None):
        if lower is not None:
            data = data[data[col_name] >= lower]
        if upper is not None:
            data = data[data[col_name] <= upper]
        return data

    def _clean_duplicate_track_ids(self, data):
        return data.drop_duplicates()

    def _feature_selection(self, data):
        return data.drop(columns=['album_name', 'track_genre', 'artists'], errors='ignore')

    def _get_track_name_sentiment(self, track_name):
        return track_name.apply(lambda x: self.sia.polarity_scores(x)['compound'])
