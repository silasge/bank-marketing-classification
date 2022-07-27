import os
from zipfile import ZipFile
import shutil
import argparse


def unzip_data(zip_file: str, member: str, to_path: str):
    with ZipFile(zip_file, "r") as z:
        member_file = z.open(member)
        member_filename = os.path.basename(member)
        member_data = open(os.path.join(to_path, member_filename), "wb")
        with member_file, member_data:
            shutil.copyfileobj(member_file, member_data)
    return member_data.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_file", type=str, dest="zip_file")
    parser.add_argument("--member", type=str, dest="member")
    parser.add_argument("--to_path", type=str, dest="to_path")
    args = parser.parse_args()
    unzip_data(zip_file=args.zip_file, member=args.member, to_path=args.to_path)