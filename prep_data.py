import os
import random
import shutil
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(
    prog="Prep Data",
    description="Splits images folder into train and test",
    epilog="Text at the bottom of help",
)

parser.add_argument("dataset_path")  # positional argument
parser.add_argument("-s", "--train_size", type=float)  # option that takes a value
parser.add_argument("-r", "--remove_source", type=bool, default=True)

args = parser.parse_args()


def split_train_test(
    dataset_path: str, train_size: int = 0.8, remove_source: bool = True
) -> None:

    try:
        Path("dataset").mkdir(parents=True, exist_ok=True)
        Path("dataset/train/class_0").mkdir(parents=True, exist_ok=True)
        Path("dataset/test/class_0").mkdir(parents=True, exist_ok=True)

    except:
        pass

    full_dataset = [os.path.join(dataset_path, i) for i in os.listdir(dataset_path)]
    random.shuffle(full_dataset)
    train_len = int(train_size * len(full_dataset))
    train_paths, test_paths = full_dataset[:train_len], full_dataset[train_len:]

    for path in train_paths:
        shutil.move(path, "dataset/train/class_0")
    for path in test_paths:
        shutil.move(path, "dataset/test/class_0")

    if remove_source:
        os.removedirs(dataset_path)


if __name__ == "__main__":
    split_train_test(args.dataset_path, args.train_size, args.remove_source)
