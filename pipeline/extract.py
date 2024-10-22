"""A script to extract data from the kaggle API"""

from os import environ as ENV, rename

import pandas as pd
from dotenv import load_dotenv
import kagglehub

SPOTIFY_DATASET = "maharshipandya/-spotify-tracks-dataset"
DOWNLOAD_PATH = "data"


def download_csv(dataset_name: str, download_path: str) -> None:
    """Returns an authenticated api"""
    load_dotenv()
    path = kagglehub.dataset_download(dataset_name)
    rename(path, download_path)


if __name__ == "__main__":
    download_csv(SPOTIFY_DATASET, DOWNLOAD_PATH)
