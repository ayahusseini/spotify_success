"""Functions for cleaning the extracted data"""
import pandas as pd

FEATURES = []


def clean_duplicate_track_ids(spotify_data: pd.DataFrame):
    """Drops duplicate rows from the dataframe"""


def clean_bounded_column(column: pd.Series, min: float, max: float | None) -> pd.Series:
    """Removes entries outside of the minimum and maximum bounds"""


def feature_selection(spotify_data: pd.DataFrame) -> pd.DataFrame:
    """Return the dataframe with only the relevant features"""


def clean_data(spotify_data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe"""
