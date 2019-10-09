import random
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from allennlp.modules.elmo import _ElmoBiLm as ElmoBiLm
from allennlp.nn.util import remove_sentence_boundaries


class RNNModel(nn.Module):
    def __init__(self, config):
        super(RNNModel, self).__init__()
        self.config = config

        self.dim_embedding = config["dim_embedding"]
        self.dim_elmo_embedding = config.get("dim_elmo_embedding", 0)
        if self.dim_elmo_embedding > 0:
            self.dim_rnn_input = self.dim_elmo_embedding
        else:
            self.dim_rnn_input = self.dim_embedding
        # self.dim_rnn_input = self.dim_embedding + self.dim_elmo_embedding
        self.dim_hidden = config["dim_hidden"]
        self.vocab_size = config["vocab_size"]
        self.label_vocab_size = config["label_vocab_size"]
        self.n_layers = config["n_layers"]
        self.dropout_embedding = config["dropout_embedding"]
        self.dropout_hidden = config["dropout_hidden"]
        self.dropout_output = config["dropout_output"]
        self.bidirectional = config["bidirectional"]
        self.n_directions = 2 if self.bidirectional else 1

        self.dropout = LockedDropout()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class SLURNN(RNNModel):
    def __init__(self, *args, **kwargs):
        super(SLURNN, self).__init__(*args, **kwargs)

        self.embedding = nn.Embedding(self.vocab_size, self.dim_embedding)
        self.rnn = nn.ModuleList([
            nn.LSTM(
                self.dim_rnn_input if i == 0 else self.dim_hidden * self.n_directions,
                self.dim_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=self.bidirectional
            )
            for i in range(self.n_layers)
        ])
        self.linear = nn.Linear(
            self.dim_hidden * self.n_directions,
            self.label_vocab_size)
        self.init_weights()

    def forward(self, inputs, positions, elmo_emb=None):
        """
        args:
            inputs: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, label_vocab_size]
        """
        bs, sl = inputs.size()

        if elmo_emb is None:
            inputs = self.embedding(inputs)
            # if elmo_emb is not None:
            #    inputs = torch.cat([inputs, elmo_emb], dim=2)
        else:
            inputs = elmo_emb

        inputs = self.dropout(inputs, self.dropout_embedding)
        last_output = inputs
        for l, rnn in enumerate(self.rnn):
            output, _ = rnn(last_output)
            if l != self.n_layers - 1:
                output = self.dropout(output, self.dropout_hidden)
            last_output = output

        output = self.dropout(output, self.dropout_output)
        """
        output = output.split(1, dim=0)
        output_per_utt = torch.stack([
            torch.stack([
                output[i].squeeze(dim=0)[s:e].max(dim=0)[0]
                if s != e else torch.zeros(output[0].size(2)).to(output[0].device)
                for s, e in zip(pos[:-1], pos[1:])
            ], dim=0)
            for i, pos in enumerate(positions)
        ], dim=0)

        conv_output, _ = self.conv_rnn(output_per_utt)
        pooled_output, _ = conv_output.max(dim=1)
        """
        pooled_output = torch.stack([
            output[i, pos[-2]:pos[-1]].max(dim=0)[0]
            if pos[-2] != pos[-1] else torch.zeros(output.size(2)).to(output.device)
            for i, pos in enumerate(positions)
        ], dim=0)
        # pooled_output, _ = output.max(dim=1)
        logits = self.linear(pooled_output)
        return logits

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)


class SLULatticeRNN(SLURNN):
    def __init__(self, *args, **kwargs):
        super(SLULatticeRNN, self).__init__(*args, **kwargs)

    def forward(self, inputs, positions, prevs, nexts, elmo_emb=None):
        """
        args:
            inputs: shape [batch_size, seq_length]

        outputs:
            logits: shape [batch_size, label_vocab_size]
        """
        bs, sl = inputs.size()

        if elmo_emb is None:
            inputs = self.embedding(inputs)
            # if elmo_emb is not None:
            #    inputs = torch.cat([inputs, elmo_emb], dim=2)
        else:
            inputs = elmo_emb

        inputs = self.dropout(inputs, self.dropout_embedding)
        last_output = inputs.split(1, dim=1)
        t_hc = t_for = 0.0
        for l, rnn in enumerate(self.rnn):
            h = torch.zeros((1, bs, rnn.hidden_size)).to(inputs.device)
            c = torch.zeros((1, bs, rnn.hidden_size)).to(inputs.device)
            this_output = []
            hs = []
            cs = []
            for t in range(sl):
                if t != 0:
                    start = time.time()
                    h = []
                    c = []
                    for b, prev in enumerate(prevs):
                        tmp_h = torch.zeros((1, rnn.hidden_size)).to(inputs.device)
                        tmp_c = torch.zeros((1, rnn.hidden_size)).to(inputs.device)
                        for idx, marginal in prev[t]:
                            tmp_h += marginal * hs[idx][b]
                            tmp_c += marginal * cs[idx][b]
                        h.append(tmp_h)
                        c.append(tmp_c)
                    h = torch.stack(h, dim=1)
                    c = torch.stack(c, dim=1)
                    """
                    h = torch.stack([
                        torch.stack([
                            marginal * hs[idx][b]
                            for idx, marginal in prev[t]
                        ], dim=0).sum(dim=0)
                        for b, prev in enumerate(prevs)
                    ], dim=1)
                    c = torch.stack([
                        torch.stack([
                            marginal * cs[idx][b]
                            for idx, marginal in prev[t]
                        ], dim=0).sum(dim=0)
                        for b, prev in enumerate(prevs)
                    ], dim=1)
                    """
                    t_hc += time.time() - start
                start = time.time()
                output, (h, c) = rnn(last_output[t], (h, c))
                t_for += time.time() - start
                hs.append(h.squeeze(0).split(1, dim=0))
                cs.append(c.squeeze(0).split(1, dim=0))
                if l != self.n_layers - 1:
                    output = self.dropout(output, self.dropout_hidden)
                this_output.append(output)
            last_output = this_output

        # print(f"h & c takes {t_hc}")
        # print(f"forward takes {t_for}")
        output = torch.stack(last_output, dim=0).squeeze(2).transpose(0, 1)
        output = self.dropout(output, self.dropout_output)
        pooled_output = torch.stack([
            output[i, pos[-2]:pos[-1]].max(dim=0)[0]
            if pos[-2] != pos[-1] else torch.zeros(output.size(2)).to(output.device)
            for i, pos in enumerate(positions)
        ], dim=0)
        # pooled_output, _ = output.max(dim=1)
        logits = self.linear(pooled_output)
        return logits

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class ELMoLM(nn.Module):
    def __init__(self, option_file, weight_file, vocab_size):
        super().__init__()
        self.elmo = ElmoBiLm(option_file, weight_file, requires_grad=True)
        self.output_dim = self.elmo.get_output_dim()
        self.output_dim_half = self.output_dim // 2
        self.decoder = nn.Linear(self.output_dim_half, vocab_size)

    def forward(self, inputs):
        bilm_output = self.elmo(inputs)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        representations = []
        for representation in layer_activations:
            r, mask = remove_sentence_boundaries(representation, mask_with_bos_eos)
            representations.append(r)

        repr_forward, repr_backward = representations[-1].split(self.output_dim_half, dim=2)
        logits_forward = self.decoder(repr_forward)
        logits_backward = self.decoder(repr_backward)

        return logits_forward, logits_backward, representations, mask
