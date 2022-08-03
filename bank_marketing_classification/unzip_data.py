import argparse
import os
import shutil
from zipfile import ZipFile

from loguru import logger


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
    parser.add_argument("zip_file", type=str)
    parser.add_argument("member", type=str)
    parser.add_argument("to_path", type=str)
    args = parser.parse_args()
    logger.info(f"Extraindo arquivo {os.path.basename(args.member)} em {args.to_path}.")
    unzip_data(zip_file=args.zip_file, member=args.member, to_path=args.to_path)
    logger.info("Extração concluída.")
