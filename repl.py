import argparse


parser = argparse.ArgumentParser(
    prog="Prep Data",
    description="Splits images folder into train and test",
    epilog="Text at the bottom of help",
)


parser.add_argument("dataset_path")  # positional argument
parser.add_argument("-t", "--train_size", type=float)  # option that takes a value
parser.add_argument("-r", "--remove_source", type=bool, default=True)

args = parser.parse_args()

print(args.dataset_path, args.train_size, args.remove_source)
