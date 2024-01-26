"""
Intent: Create the pickle file from the skeleton folder and the label folder (Needed to be flatten before use)
Author: Tom
Last update: 2023/09/20
"""
import os
import pickle
from argparse import ArgumentParser, Namespace
from sklearn.model_selection import train_test_split

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/home/tom/sporttech/skating_coach/datasets/Axel2_pickle", help="Path to the data folder"
    )
    parser.add_argument(
        "--data_file", type=str, default="data.pkl", help="Filename of the pickle file"
    )
    parser.add_argument(
        "--output_train", type=str, default="train.pkl", help="Filename of the output training dataset"
    )
    parser.add_argument(
        "--output_test", type=str, default="test.pkl", help="Filename of the output testing dataset"
    )
    args = parser.parse_args()
    return args


def main(args):
    # Extract args
    data_dir = args.data_dir
    filename = os.path.join(data_dir, args.data_file)
    output_train = os.path.join(data_dir, args.output_train)
    output_test = os.path.join(data_dir, args.output_test)

    with open(filename, 'rb') as file:
        data = pickle.load(file)
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    print(f'Total dataset size: {len(data)}')
    print(f'-' * len(f'Total dataset size: {len(data)}'))
    print(f'Train size: {len(train_data)}')
    print(f'Test size: {len(test_data)}')
    
    # Save files
    with open(output_train, 'wb') as file:
        pickle.dump(train_data, file)
    with open(output_test, 'wb') as file:
        pickle.dump(test_data, file)

if __name__ == "__main__":
    args = parse_args()
    main(args)