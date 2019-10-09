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
from dataset import LMDataset, ConfusionDataset
from model import LM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help="model directory")
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--ca_finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
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


def ca_finetune(args):
    config = load_config(args.model_dir)

    train_dataset = ConfusionDataset(
        config["train_file"],
        vocab_file=config["vocab_file"],
        stop_word_file=config.get("stop_word_file", None))

    vocab_dump_path = os.path.join(args.model_dir, "vocab.pkl")
    with open(vocab_dump_path, 'wb') as fp:
        pickle.dump(train_dataset.vocab, fp)

    valid_dataset = ConfusionDataset(
        config["valid_file"],
        vocab_dump=vocab_dump_path,
        stop_word_file=config.get("stop_word_file", None))

    config["vocab_size"] = len(train_dataset.vocab)
    model = LM(config, args.model_dir)

    if args.epoch is not None:
        print_time_info("Loading checkpoint {} from model_dir".format(args.epoch))
        model.load_model(args.model_dir, args.epoch)

    model.train(
        epochs=config["train_epochs"],
        batch_size=config["batch_size"],
        data_engine=train_dataset,
        valid_data_engine=valid_dataset,
        train_decoder_epochs=config.get("train_decoder_epochs", 0),
        max_iter_per_epoch=config.get("max_iter_per_epoch", 100000)
    )


def train(args):
    config = load_config(args.model_dir)

    train_dataset = LMDataset(
        config["train_file"],
        vocab_file=config["vocab_file"])

    vocab_dump_path = os.path.join(args.model_dir, "vocab.pkl")
    with open(vocab_dump_path, 'wb') as fp:
        pickle.dump(train_dataset.vocab, fp)

    valid_dataset = LMDataset(
        config["valid_file"],
        vocab_dump=vocab_dump_path)

    config["vocab_size"] = len(train_dataset.vocab)
    model = LM(config, args.model_dir)

    if args.epoch is not None:
        print_time_info("Loading checkpoint {} from model_dir".format(args.epoch))
        model.load_model(args.model_dir, args.epoch)

    model.train(
        epochs=config["train_epochs"],
        batch_size=config["batch_size"],
        data_engine=train_dataset,
        valid_data_engine=valid_dataset,
        train_decoder_epochs=config.get("train_decoder_epochs", 0),
        max_iter_per_epoch=config.get("max_iter_per_epoch", 100000)
    )


def test(args):
    config = load_config(args.model_dir)

    vocab_dump_path = os.path.join(args.model_dir, "vocab.pkl")

    test_file = config["test_file"] if len(args.test_file) == 0 else args.test_file
    test_dataset = LMDataset(
        test_file,
        vocab_dump=vocab_dump_path)

    config["vocab_size"] = len(test_dataset.vocab)
    model = LM(config, args.model_dir)

    if args.epoch is not None:
        print_time_info("Loading checkpoint {} from model_dir".format(args.epoch))
        epoch = model.load_model(args.model_dir, args.epoch)
    else:
        print_time_info("Loading last checkpoint from model_dir")
        epoch = model.load_model(args.model_dir)

    loss = model.test(
        batch_size=config["batch_size"],
        data_engine=test_dataset
    )


if __name__ == "__main__":
    args = get_args()
    # with ipdb.launch_ipdb_on_exception():
    #     sys.breakpointhook = ipdb.set_trace

    if args.ca_finetune:
        ca_finetune(args)
    elif args.test:
        test(args)
    else:
        train(args)
