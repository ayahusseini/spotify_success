# Pipeline

## Setup 

1. Setup virtual environment
```sh
python3 -m venv venv
```

2. Download requirements 
```sh
pip install -r requirements.txt
```

3. Setup environment variables 
```sh
KAGGLE_USERNAME=XXXXXXX
KAGGLE_KEY=XXXXXXXXXXXXXXXX
```

## Files 

- `extract.py` downloads the dataset and returns it as a pandas dataframe
- `data_cleaning.py` contains useful functions for data cleaning
- `split_test_train.py` contains functions for splitting the dataset into testing and training data.


## Implementation details 

### Test set creation 
To avoid a data snooping bias, a test set should be created from the extracted data. This program should always create the same test set given the same initial data. In other words, if a track is in the test set once, then it should be in all future test sets. Otherwise, eventually the ML algorithm will be trained on all tracks. There are different options for doing this:

- Save the test set on the first run and reload it in subsequent runs 
- Set the random number generator's seed `np.random.seed()` so that the same 'random' data is always selected.

The problem with the above two methods is that they break if the initial dateset is updated or changed. 

Instead, we use hashing to ensure that the test set remains consistent even if the dataset is changed. Each track has a unique `track_id`. Each `track_id` is a 22-character length string. 
