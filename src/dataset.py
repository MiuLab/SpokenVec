import pickle
import os
import json
import csv
import re

import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm

from vocabulary import Vocab, PAD
from model_utils import pad_sequences, index_to_nhot
from utils import print_time_info
from lattice_utils import LatticeReader


class SLUDataset(Dataset):
    label_idx = 3

    def __init__(self, filename, vocab_file=None,
                 vocab_dump=None, label_vocab_dump=None,
                 n_prev_turns=0):
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            self.data = [row for row in reader]
        if "id" in self.data[0]:
            self.id2idx = {row["id"]: i for i, row in enumerate(self.data)}
        self.n_prev_turns = n_prev_turns
        if vocab_dump is None:
            self.vocab = Vocab(vocab_file)
        else:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)
        if label_vocab_dump is None:
            labels = [row["label"] for row in self.data]
            self.label_vocab = LabelVocab(labels)
        else:
            with open(label_vocab_dump, 'rb') as fp:
                self.label_vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _process_text(self, text):
        for punct in [',', '.', '?', '!']:
            if text.endswith(f" {punct}"):
                text = text[:-2]
        text = re.sub(" ([a-z])\. ", " \\1 ", text)
        return text

    def collate_fn(self, batch):
        inputs, words, positions, labels = [], [], [], []
        for utt in batch:
            prev_utts = []
            poss = [0]
            prev_id = utt.get("previous", "")
            while len(prev_utts) < self.n_prev_turns and prev_id != "":
                if prev_id not in self.id2idx:
                    break
                prev_row = self.data[self.id2idx[prev_id]]
                prev_id = prev_row["previous"]
                text = self._process_text(prev_row["text"])
                prev_utts = [text] + prev_utts

            text = self._process_text(utt["text"])
            for prev_utt in prev_utts:
                poss.append(poss[-1] + len(prev_utt.split()))
            poss.append(poss[-1] + len(text.split()))
            while len(poss) - 1 < self.n_prev_turns + 1:
                poss = [0] + poss
            label = utt["label"]
            text = " ".join(prev_utts + [text])
            word_ids = [self.vocab.w2i(word) for word in text.split()]
            words.append(text.split())
            inputs.append(word_ids)
            positions.append(poss)
            labels.append(self.label_vocab.l2i(label))

        max_length = max(map(len, inputs))
        inputs = pad_sequences(inputs, max_length)
        labels = np.array(labels)
        return inputs, words, positions, labels


class SLULatticeDataset(Dataset):
    label_idx = 5

    def __init__(self, filename, vocab_file=None,
                 vocab_dump=None, label_vocab_dump=None,
                 n_prev_turns=0, text_input=False):
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            self.data = [row for row in reader]
            lattice_reader = LatticeReader(text_input=text_input)
            for i, row in enumerate(tqdm(self.data)):
                row["lattice"] = lattice_reader.read_sent(row["text"], i)

        self.id2idx = {row["id"]: i for i, row in enumerate(self.data)}
        self.n_prev_turns = n_prev_turns
        if vocab_dump is None:
            self.vocab = Vocab(vocab_file)
        else:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)
        if label_vocab_dump is None:
            labels = [row["label"] for row in self.data]
            self.label_vocab = LabelVocab(labels)
        else:
            with open(label_vocab_dump, 'rb') as fp:
                self.label_vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _process_text(self, text):
        for punct in [',', '.', '?', '!']:
            if text.endswith(f" {punct}"):
                text = text[:-2]
        text = re.sub(" ([a-z])\. ", " \\1 ", text)
        return text

    def collate_fn(self, batch):
        inputs, words, positions, prevs, nexts, labels = [], [], [], [], [], []
        for utt in batch:
            text = " ".join(utt['lattice'].str_tokens())
            label = utt["label"]
            prev = []
            next = []
            for node in utt['lattice'].nodes:
                prev.append([
                    (n, np.exp(utt['lattice'].nodes[n].marginal_log_prob))
                    for n in node.nodes_prev
                ])
                next.append([
                    (n, np.exp(utt['lattice'].nodes[n].fwd_log_prob))
                    for n in node.nodes_next
                ])
            word_ids = [self.vocab.w2i(word) for word in text.split()]
            words.append(text.split())
            inputs.append(word_ids)
            positions.append([0, len(word_ids)])
            prevs.append(prev)
            nexts.append(next)
            labels.append(self.label_vocab.l2i(label))

        max_length = max(map(len, inputs))
        inputs = pad_sequences(inputs, max_length)
        prevs = [
            prev + [[(-1, 0.0)] for _ in range(max_length-len(prev))]
            for prev in prevs
        ]
        nexts = [
            next + [[(-1, 0.0)] for _ in range(max_length-len(next))]
            for next in nexts
        ]
        labels = np.array(labels)
        return inputs, words, positions, prevs, nexts, labels


class LabelVocab:
    def __init__(self, labels):
        self.build_vocab(labels)

    def build_vocab(self, labels):
        unique_labels = set(labels)
        sorted_labels = sorted(list(unique_labels))
        self.vocab = sorted_labels
        self.rev_vocab = dict()
        for i, label in enumerate(sorted_labels):
            self.rev_vocab[label] = i

    def l2i(self, label):
        try:
            return self.rev_vocab[label]
        except:
            raise KeyError(label)

    def i2l(self, index):
        return self.vocab[index]


class ConfusionDataset(Dataset):
    def __init__(self, filename, vocab_file=None, vocab_dump=None,
                 stop_word_file=None):
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]

        self.stop_words = set()
        if stop_word_file is not None:
            for line in open(stop_word_file):
                self.stop_words.add(line.strip())

        datas = []
        count, total = 0, 0
        for row in data:
            ref = row["transcription"]
            hyp = row["hypothesis"]
            score = float(row["score"])
            confs = row["confusion"].split()
            confs = [
                (confs[i*3], confs[i*3+1])
                for i in range(len(confs)//3+1)
            ]
            conf_ids = []
            ref_id = hyp_id = 0
            for ref_w, hyp_w in confs:
                ref_eps = (ref_w == "<eps>")
                hyp_eps = (hyp_w == "<eps>")
                if not ref_eps and not hyp_eps and ref_w != hyp_w:
                    total += 1
                    if ref_w not in self.stop_words and hyp_w not in self.stop_words:
                        conf_ids.append((ref_id, hyp_id))
                    else:
                        count += 1

                if not ref_eps:
                    ref_id += 1
                if not hyp_eps:
                    hyp_id += 1
            datas.append((ref, hyp, conf_ids, score))
        print(count, total)
        self.data = datas

        if vocab_file is not None:
            self.vocab = Vocab(vocab_file)
        elif vocab_dump is not None:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch):
        transposed = zip(*batch)
        return tuple(transposed)

    def collate_fn_atis(self, batch):
        refs, ref_out_for, ref_out_rev = [], [], []
        hyps, hyp_out_for, hyp_out_rev = [], [], []
        confs, scores = [], []
        for ref, hyp, conf_ids, score in batch:
            ref = ref.strip().split()
            hyp = hyp.strip().split()
            confs.append(conf_ids)
            scores.append(score)
            refs.append(ref)
            ref_word_ids = [self.vocab.w2i(w) for w in ref]
            ref_out_for.append(ref_word_ids[1:] + [PAD])
            ref_out_rev.append([PAD] + ref_word_ids[:-1])
            hyps.append(hyp)
            hyp_word_ids = [self.vocab.w2i(w) for w in hyp]
            hyp_out_for.append(hyp_word_ids[1:] + [PAD])
            hyp_out_rev.append([PAD] + hyp_word_ids[:-1])

        inputs = refs + hyps
        outputs_for = ref_out_for + hyp_out_for
        outputs_rev = ref_out_rev + hyp_out_rev
        max_length = max([len(sent) for sent in outputs_for])

        outputs_for = pad_sequences(outputs_for, max_length, 'post')
        outputs_rev = pad_sequences(outputs_rev, max_length, 'post')

        return inputs, outputs_for, outputs_rev, confs, scores


class LMDataset(Dataset):
    def __init__(self, text_path, vocab_file=None, vocab_dump=None):
        self.data = []

        print_time_info("Reading text from {}".format(text_path))

        with open(text_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                words = row["text"].split()
                if "id" in row:
                    self.data.append((row["id"], words))
                else:
                    self.data.append((i, words))
        # for line in tqdm(open(text_path)):
        #     uid, *words = line.strip().split()
        #     self.data.append((uid, words))

        if vocab_dump is None:
            self.vocab = Vocab(vocab_file)
        else:
            with open(vocab_dump, 'rb') as fp:
                self.vocab = pickle.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uid, sentence = self.data[index]
        word_ids = [self.vocab.w2i(word) for word in sentence]
        return uid, sentence, word_ids

    def collate_fn(self, batch):
        uids, inputs, outputs, outputs_rev = [], [], [], []
        for uid, words, word_ids in batch:
            uids.append(uid)
            inputs.append(words)
            outputs.append(word_ids[1:] + [PAD])
            outputs_rev.append([PAD] + word_ids[:-1])

        max_length = max([len(sent) for sent in outputs])
        # (batch_size, seq_length)
        outputs = pad_sequences(outputs, max_length, 'post')
        outputs_rev = pad_sequences(outputs_rev, max_length, 'post')

        return inputs, outputs, outputs_rev, uids


if __name__ == "__main__":
    dataset = ConfusionDataset('../data/csv/dev-asr-conf.csv')
    print(len(dataset))
    for i in range(2000, 2010):
        print(dataset[i])
