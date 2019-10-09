import json

from tqdm import tqdm

from utils import print_time_info


BOS = 0
EOS = 1
PAD = 2
BOS_SYMBOL = '<BOS>'
EOS_SYMBOL = '<EOS>'
PAD_SYMBOL = '<PAD>'

class Vocab:
    def __init__(self, vocab_path):
        print_time_info("Reading vocabulary from {}".format(vocab_path))
        self.read_vocab(vocab_path)

    def read_vocab(self, vocab_path):
        self.vocab = dict()
        self.rev_vocab = dict()

        self.add_word(BOS_SYMBOL)
        self.add_word(EOS_SYMBOL)
        self.add_word(PAD_SYMBOL)
        self.bos_symbol = BOS_SYMBOL
        self.eos_symbol = EOS_SYMBOL
        self.pad_symbol = PAD_SYMBOL

        vocabs = set()
        for wid, line in tqdm(enumerate(open(vocab_path))):
            word = line.strip()
            vocabs.add(word)

        if '<unk>' in vocabs:
            self.unk_symbol = '<unk>'
        elif '<UNK>' in vocabs:
            self.unk_symbol = '<UNK>'
        else:
            self.unk_symbol = '<unk>'
        self.add_word(self.unk_symbol)

        for word in sorted(vocabs):
            self.add_word(word)

    def w2i(self, word):
        if word not in self.vocab:
            word = self.unk_symbol
        return self.vocab[word]

    def i2w(self, index):
        return self.rev_vocab[index]

    def add_word(self, word):
        if word in self.vocab:
            return
        wid = len(self.vocab)
        self.vocab[word] = wid
        self.rev_vocab[wid] = word

    def __len__(self):
        return len(self.vocab)
