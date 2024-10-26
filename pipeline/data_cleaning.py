"""Functions for cleaning the extracted data"""
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")


def clean_bounded_column(data:pd.DataFrame,col_name: str,lower:float|None,upper:float|None) -> bool:
    """Returns True if a quantity is within the lower and upper bounds.
    If a bound is set to None, it is assumed to not exist."""
    if lower is not None:
        data = data[data[col_name] >= lower]
    if upper is not None:
        data = data[data[col_name] <= upper]
    return data


def clean_duplicate_track_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Drops duplicate rows from the dataframe"""
    data = data.drop_duplicates()
    return data

def feature_selection(data: pd.DataFrame) -> pd.DataFrame:
    """Return the dataframe with only the relevant features"""
    data = data.drop(labels = ['album_name','track_genre', 'artists'], axis=1)
    return data 

def get_track_name_sentiment(track_name: pd.Series) -> pd.Series:
    """Return a series containing the sentiment of each track name"""
    sia = SentimentIntensityAnalyzer()
    return track_name.apply(lambda x: sia.polarity_scores(x)['compound'])

def clean_data(spotify_data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe"""
    spotify_data = spotify_data.dropna()

    spotify_data = feature_selection(spotify_data)
    spotify_data = clean_duplicate_track_ids(spotify_data)

    spotify_data = clean_bounded_column(spotify_data, 'popularity', lower=0, upper=100)

    for col in ['duration_ms','loudness','tempo']:
        spotify_data = clean_bounded_column(spotify_data, 'duration_ms', lower=0, upper=None)

    for col in ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']:
        spotify_data = clean_bounded_column(spotify_data, col, lower=0, upper=1)

    spotify_data = spotify_data[spotify_data['key'] in range(-1,12)]
    spotify_data = spotify_data[spotify_data['time_signature'] in range(3,8)]
    spotify_data = spotify_data[spotify_data['mode'] in (0,1)]


    spotify_data['track_name_sentiment'] = get_track_name_sentiment(spotify_data['track_name'])
    
    return spotify_data

