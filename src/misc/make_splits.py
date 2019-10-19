import argparse
import json
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_fold', type=int)
    parser.add_argument('n_data', type=int)
    parser.add_argument('output')
    args = parser.parse_args()
    return args


def make_splits(n_fold, n_data, output):
    indices = list(range(n_data))
    random.shuffle(indices)
    splits = []
    folds = [
        indices[int(n_data*(i/n_fold)):int(n_data*((i+1)/n_fold))]
        for i in range(n_fold)
    ]
    assert n_data == sum([len(fold) for fold in folds])

    for i in range(n_fold):
        valid_indices = folds[i % n_fold]
        test_indices = folds[(i+1) % n_fold]
        train_indices = []
        for j in range(n_fold):
            if j != (i % n_fold) and j != ((i+1) % n_fold):
                train_indices += folds[j]
        splits.append({
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "test_indices": test_indices
        })

    with open(output, 'w') as fw:
        json.dump(splits, fw, indent=4)


if __name__ == "__main__":
    args = get_args()
    make_splits(args.n_fold, args.n_data, args.output)
