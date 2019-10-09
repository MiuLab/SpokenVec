import argparse
import sys
import json
import os
import ipdb
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import print_time_info
from dataset import SLUDataset, SLULatticeDataset
from model import SLU


DATASETS = {
    'text': SLUDataset,
    'lattice': SLULatticeDataset
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help="model directory")
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--test_file', type=str, default="")
    args = parser.parse_args()

    return args


def load_config(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "config.json not present in '{}'".format(model_dir))

    with open(config_path) as f:
        config = json.load(f)

    return config


def train(args):
    config = load_config(args.model_dir)
    dataset_cls = DATASETS[config.get("dataset_cls", "text")]
    train_dataset = dataset_cls(
        config["train_file"],
        vocab_file=config["vocab_file"],
        label_vocab_dump=config.get("label_vocab_dump", None),
        n_prev_turns=config.get("n_prev_turns", 0),
        **(config.get("dataset_args", {})))

    vocab_dump_path = os.path.join(args.model_dir, "vocab.pkl")
    with open(vocab_dump_path, 'wb') as fp:
        pickle.dump(train_dataset.vocab, fp)
    label_vocab_dump_path = os.path.join(args.model_dir, "label_vocab.pkl")
    with open(label_vocab_dump_path, 'wb') as fp:
        pickle.dump(train_dataset.label_vocab, fp)

    valid_dataset = dataset_cls(
        config["valid_file"],
        vocab_dump=vocab_dump_path,
        label_vocab_dump=label_vocab_dump_path,
        n_prev_turns=config.get("n_prev_turns", 0),
        **(config.get("dataset_args", {})))

    test_dataset = None
    if len(args.test_file) > 0:
        test_dataset = dataset_cls(
            args.test_file,
            vocab_dump=vocab_dump_path,
            label_vocab_dump=label_vocab_dump_path,
            n_prev_turns=config.get("n_prev_turns", 0),
            **(config.get("dataset_args", {})))

    config["model"]["vocab_size"] = len(train_dataset.vocab)
    config["model"]["label_vocab_size"] = len(train_dataset.label_vocab.vocab)
    model = SLU(config, args.model_dir)

    if args.epoch is not None:
        print_time_info("Loading checkpoint {} from model_dir".format(args.epoch))
        model.load_model(args.model_dir, args.epoch)

    model.train(
        epochs=config["train_epochs"],
        batch_size=config["batch_size"],
        data_engine=train_dataset,
        valid_data_engine=valid_dataset,
        test_data_engine=test_dataset
    )


def test(args):
    config = load_config(args.model_dir)
    dataset_cls = DATASETS[config.get("dataset_cls", "text")]

    vocab_dump_path = os.path.join(args.model_dir, "vocab.pkl")
    label_vocab_dump_path = os.path.join(args.model_dir, "label_vocab.pkl")

    test_file = config["test_file"] if len(args.test_file) == 0 else args.test_file
    test_dataset = dataset_cls(
        test_file,
        vocab_dump=vocab_dump_path,
        label_vocab_dump=label_vocab_dump_path,
        n_prev_turns=config.get("n_prev_turns", 0),
        **(config.get("dataset_args", {})))

    config["model"]["vocab_size"] = len(test_dataset.vocab)
    config["model"]["label_vocab_size"] = len(test_dataset.label_vocab.vocab)
    model = SLU(config, args.model_dir)

    if args.epoch is not None:
        print_time_info("Loading checkpoint {} from model_dir".format(args.epoch))
        epoch = model.load_model(args.model_dir, args.epoch)
    else:
        print_time_info("Loading last checkpoint from model_dir")
        epoch = model.load_model(args.model_dir)

    loss, acc, y_true, y_pred = model.test(
        batch_size=config["batch_size"],
        data_engine=test_dataset,
        report=True,
        verbose=args.verbose
    )


if __name__ == "__main__":
    args = get_args()
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace

        if args.finetune:
            finetune(args)
        elif args.test:
            test(args)
        else:
            train(args)
