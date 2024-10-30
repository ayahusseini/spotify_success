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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data"""
        X = X.dropna()

        X = self._feature_selection(X)

        X = self._clean_duplicate_track_ids(X)

        X = self._clean_bounded_column(
            X, 'popularity', self.popularity_lower, self.popularity_upper)

        for col in ['duration_ms', 'tempo']:
            X = self._clean_bounded_column(X, col, lower=0, upper=None)

        for col in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']:
            X = self._clean_bounded_column(X, col, lower=0, upper=1)

        X = X[X['key'].isin(range(-1, 12))]

        X = X[X['time_signature'].isin(range(3, 8))]

        X = X[X['mode'].isin([0, 1])]

        return X

    def _clean_bounded_column(self, data, col_name, lower=None, upper=None):
        """Cleans out-of-bounds data"""
        if lower is not None:
            data = data[data[col_name] >= lower]
        if upper is not None:
            data = data[data[col_name] <= upper]
        return data

    def _clean_duplicate_track_ids(self, data):
        """Removes duplicate track ids"""
        return data.drop_duplicates()

    def _feature_selection(self, data):
        """Drops irrelevant features"""
        return data.drop(columns=['album_name', 'track_genre', 'artists'], errors='ignore')


class AttributeAdder(BaseEstimator, TransformerMixin):
    """Custom transformer to add attributes to the dataframe"""

    def __init__(self, add_track_name_sentiment: bool = True, add_danceability_to_speechiness: bool = True):
        self.add_track_name_sentiment = add_track_name_sentiment
        self.add_danceability_to_speechiness = add_danceability_to_speechiness

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.add_track_name_sentiment:
            X['track_name_sentiment'] = self._get_track_name_sentiment(
                X['track_name'])

        if self.add_danceability_to_speechiness:
            X['danceability_to_speechiness'] = X['danceability'] / X['speechiness']

        return X

    def _get_track_name_sentiment(self, track_name: pd.Series) -> pd.Series:
        """Returns a series containing sentiment score for each track name"""
        return track_name.apply(lambda x: self.sia.polarity_scores(x)['compound'])
