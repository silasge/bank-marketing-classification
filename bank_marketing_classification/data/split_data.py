import os
import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split


def split_data(csv_file: str, test_size: float=0.25, random_state: int=42):
    data = pd.read_csv(csv_file, sep=";")
    return train_test_split(data, test_size=test_size, random_state=random_state)


def export_split_data(train_test_data: list, save_to: str):
    data_train, data_test = train_test_data
    if "train" in save_to:
        data_train.to_csv(save_to, index=False)
    elif "test" in save_to:
        data_test.to_csv(save_to, index=False)
    return data_train, data_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, dest="csv_file")
    parser.add_argument("--test_size", type=float, dest="test_size")
    parser.add_argument("--save_to", type=str, dest="save_to")
    args = parser.parse_args()
    train_test_data = split_data(csv_file=args.csv_file, test_size=args.test_size)
    export_split_data(train_test_data, save_to=args.save_to)