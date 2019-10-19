import argparse
import csv
import glob
import os
import re
import tempfile
import subprocess

import editdistance as ed

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="csv file containing dataset")
    parser.add_argument("asr", help="asr transcription")
    parser.add_argument("output", help="csv file to write the asr dataset")
    args = parser.parse_args()
    return args


def read_trans(asr_file):
    fd, name = tempfile.mkstemp()
    with open(name, "w") as fw:
        subprocess.call(['sed', '-f', '/home/cwhuang/kaldi/egs/aspire/s5/local/wer_output_filter', asr_file], stdout=fw)
    processed = dict()
    for line in open(name):
        try:
            key, text = line.strip().split(maxsplit=1)
        except ValueError:
            key, text = line.strip(), ""
        processed[key] = text

    os.close(fd)
    os.remove(name)
    return processed


def make_real_trans(dataset):
    fd, name = tempfile.mkstemp()
    with open(name, 'w') as fw:
        for row in dataset:
            id, text = row["id"], row["text"]
            fw.write(f"{id} {text}\n")

    fd2, name2 = tempfile.mkstemp()
    with open(name2, "w") as fw:
        subprocess.call(['sed', '-f', '/home/cwhuang/kaldi/egs/aspire/s5/local/wer_output_filter', name], stdout=fw)
    processed = dict()
    for line in open(name2):
        try:
            key, text = line.strip().split(maxsplit=1)
        except ValueError:
            key, text = line.strip(), ""
        processed[key] = text

    os.close(fd)
    os.remove(name)
    os.close(fd2)
    os.remove(name2)
    return processed


def build(args):
    with open(args.csv) as csvfile:
        reader = csv.DictReader(csvfile)
        dataset = [row for row in reader]

    asr_trans = read_trans(args.asr)
    real_trans = make_real_trans(dataset)
    asr_dataset = []
    errors, total = 0, 0
    for row in dataset:
        id = row["id"]
        if id not in asr_trans:
            continue
        trans = asr_trans[id]
        ans_trans = real_trans[id]
        errors += ed.eval(trans.split(), ans_trans.split())
        total += len(ans_trans.split())
        row["text"] = trans
        asr_dataset.append(row)

    with open(args.output, 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            ["id", "text", "label"]
        )
        writer.writeheader()
        writer.writerows(asr_dataset)

    print(f"WER: {errors/total} ({errors}/{total})")


if __name__ == "__main__":
    args = get_args()
    build(args)
