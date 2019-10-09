import argparse
import os
import csv
import ast


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, nargs='+',
                        help="files to be read")
    parser.add_argument('output', type=str)
    parser.add_argument('--min_count', type=int, default=1)
    args = parser.parse_args()

    return args


def add_sentence(wc, sentence):
    for word in sentence.strip().split():
        if not word in wc:
            wc[word] = 0
        wc[word] += 1


def read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        data = [" ".join([item[0] for item in ast.literal_eval(row["text"])[0]]) for row in reader]
    return data


if __name__ == "__main__":
    args = get_args()

    if isinstance(args.files, str):
        files = [args.files]
    else:
        files = args.files

    word_counts = dict()
    for file in files:
        data = read_csv(file)
        for sentence in data:
            add_sentence(word_counts, sentence)

    words_sorted = sorted(word_counts.items(), key=lambda x: -x[1])
    words = [tup[0] for tup in words_sorted if tup[1] >= args.min_count]
    with open(args.output, 'w') as fw:
        for word in words:
            fw.write("{}\n".format(word))
