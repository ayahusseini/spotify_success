"""A script to extract data from the kaggle API"""

from os import rename, remove
import os
import pandas as pd
from dotenv import load_dotenv
import kagglehub

SPOTIFY_DATASET = "maharshipandya/-spotify-tracks-dataset"
DOWNLOAD_PATH = "dataset.csv"


def download_csv(dataset_name: str, download_path: str) -> None:
    """Downloads the dataset as a CSV file"""
    load_dotenv()
    path = kagglehub.dataset_download(
        dataset_name, force_download=True)
    path += f"/{os.listdir(path)[0]}"
    rename(path, download_path)


def load_csv_as_df(filepath: str) -> pd.DataFrame:
    """Loads a CSV file as a dataframe"""
    return pd.read_csv(filepath)


def extract(dataset: str, download_filepath: str, delete_csv: bool) -> pd.DataFrame:
    """Extracts the data as a dataframe.
    Optionally, deletes the CSV downloaded"""
    download_csv(dataset, download_filepath)
    data = load_csv_as_df(download_filepath)
    if delete_csv:
        remove(download_filepath)
    return data


if __name__ == "__main__":
    data = extract(SPOTIFY_DATASET, DOWNLOAD_PATH, True)
    print(data.columns)
