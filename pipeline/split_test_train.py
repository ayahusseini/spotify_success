"""Functions for creating test and train datasets"""
import hashlib
import pandas as pd 

def is_in_test(track_id:str, test_ratio:float = 0.2) -> bool:
    """Returns True if a track_id is in the test set"""
    hash_val = int(hashlib.sha256(track_id.encode()).hexdigest(),16)
    return hash_val  < test_ratio * 2**256

def perform_test_train_split(data:pd.DataFrame) -> tuple:
    """Returns the test set and the train set"""
    is_test = data["track_id"].apply(is_in_test)
    return data[is_test], data[~is_test]

if __name__ == "__main__":
    sample_data = pd.DataFrame({
    'track_id': ['abc123xyz789', 'def456uvw012', 'ghi789rst345', 'jkl012opq678','6rqhFgbbKwnb9MLmUQDhG6']
    })
    test, train = perform_test_train_split(sample_data)
    print(f"original length = {len(sample_data)}")
    print(f"test set length: {len(test)}")
    print(f"train set length: {len(train)}")
