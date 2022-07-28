import os
import argparse
from typing import Optional
import pandas as pd 
from sklearn.model_selection import train_test_split


def split_data(
    csv_file: str, 
    test_size: float=0.25, 
    random_state: int=42, 
    save_to: Optional[str] = None
):
    data = pd.read_csv(csv_file, sep=";")
    data_train, data_test = train_test_split(data, test_size=test_size, random_state=random_state)
    if save_to:
        data_train.to_csv(os.path.join(save_to, "bank_train.csv"), index=False)
        data_test.to_csv(os.path.join(save_to, "bank_test.csv"), index=False)
    return data_train, data_test

def start():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str)
    parser.add_argument("save_to", type=str)
    parser.add_argument("--test_size", type=float, default=0.25, dest="test_size")
    parser.add_argument("--random_state", type=int, default=42, dest="random_state")
    args = parser.parse_args()
    split_data(
        csv_file=args.csv_file,
        test_size=args.test_size,
        random_state=args.random_state,
        save_to=args.save_to
    )