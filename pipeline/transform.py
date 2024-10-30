"""Script for transforming the extracted data."""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from extract import extract, SPOTIFY_DATASET, DOWNLOAD_PATH
from split_test_train import perform_test_train_split
from data_cleaning import InvalidDataCleaner, AttributeAdder


def create_preprocessing_pipeline(categorical_features: list[str], numerical_features: list[str]) -> Pipeline:
    """Creates and returns a pipeline for cleaning the invalid values, scaling the numerical attributes, and encoding the categorical attributes."""

    base_pipeline = Pipeline([
        ('data_cleaner', InvalidDataCleaner(
            popularity_lower=0, popularity_upper=100)),
        ('attrib_adder', AttributeAdder(
            add_danceability_to_speechiness=True, add_track_name_sentiment=True))
    ])

    preprocessor = ColumnTransformer([
        ('num_scaling', StandardScaler(), numerical_features),
        ('cat_encoding', OneHotEncoder(), categorical_features)
    ])

    return Pipeline([
        ('clean_and_add_attributes', base_pipeline),
        ('preprocess', preprocessor)
    ])


def preprocess_data(data):
    """Transforms the extracted train data."""
    categorical_attributes = ['key', 'time_signature', 'mode']
    numerical_attributes = ["popularity", "duration_ms", "danceability", "energy", "loudness", "speechiness",
                            "instrumentalness", "liveness", "valence", "tempo", "track_name_sentiment"]
    pipeline = create_preprocessing_pipeline(
        categorical_attributes, numerical_attributes)

    return pipeline.fit_transform(data)


if __name__ == "__main__":
    data = extract(SPOTIFY_DATASET, DOWNLOAD_PATH)
    test, train = perform_test_train_split(data)
    preprocessed = preprocess_data(train)
