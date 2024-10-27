"""Tests the split_test_train.py file"""
import pandas as pd 
from split_test_train import perform_test_train_split


def test_test_train_split_consistency(sample_data):

    test_set_1, train_set_1 = perform_test_train_split(sample_data)
    test_set_2, train_set_2 = perform_test_train_split(sample_data)

    assert set(test_set_1["track_id"]) == set(test_set_2["track_id"]), "Inconsistent test set across runs"
    assert set(train_set_1["track_id"]) == set(train_set_2["track_id"]), "Inconsistent train set across runs"
