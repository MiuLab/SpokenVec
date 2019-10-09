import numpy as np
import torch

from torch import optim

from vocabulary import PAD


def pad_sequences(data, max_length, pad_type="post"):
    padded_data = np.full((len(data), max_length), PAD)

    if pad_type == "post":
        for idx, d in enumerate(data):
            padded_data[idx][:min(max_length, len(d))] = \
                    d[:min(max_length, len(d))]
    elif pad_type == "pre":
        for idx, d in enumerate(data):
            padded_data[idx][max(0, max_length-len(d)):] = \
                    d[:min(max_length, len(d))]
    return padded_data


def index_to_nhot(data, vocab_size):
    matrix = np.zeros((len(data), vocab_size), dtype=np.float32)
    for i, item in enumerate(data):
        matrix[i, item] = 1

    return matrix


def precision(log_preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = (log_preds > thresh).float()
    tpos = torch.mul((targs == pred_pos).float(), targs.float())
    return (tpos.sum()/(pred_pos.sum() + epsilon))


def recall(log_preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = (log_preds > thresh).float()
    tpos = torch.mul((targs == pred_pos).float(), targs.float())
    return (tpos.sum()/(targs.sum() + epsilon))


def f1(log_preds, targs, thresh=0.5, epsilon=1e-8):
    p = precision(log_preds, targs, thresh, epsilon)
    r = recall(log_preds, targs, thresh, epsilon)
    if p == 0 or r == 0:
        return torch.tensor(0.0)
    return (2 * p * r) / (p + r)


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_optimizer(optimizer, parameters, learning_rate):
    if optimizer.lower() == "adam":
        return optim.Adam(
                parameters, lr=learning_rate)
    elif optimizer.lower() == "rmsprop":
        return optim.RMSprop(
                parameters, lr=learning_rate)
    elif optimizer.lower() == "sgd":
        return optim.SGD(
                parameters, lr=learning_rate)
