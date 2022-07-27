import os
from urllib.request import urlretrieve 
import argparse


def download_data(url: str, to_path: str) -> tuple:
    file_name = os.path.basename(url)
    download_path = os.path.join(to_path, file_name)
    return urlretrieve(url, download_path)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, dest="url")
    parser.add_argument("--to_path", type=str, dest="to_path")
    args = parser.parse_args()
    download_data(url=args.url, to_path=args.to_path)