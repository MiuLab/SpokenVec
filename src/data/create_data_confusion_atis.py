import argparse
import csv
import subprocess
import os
import tempfile
import math

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    return args


def read_csv(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]
    return data


def read_stop_words():
    ret = set()
    for line in open("data/stop_words.txt"):
        ret.add(line.strip())
    return ret


def process_hyp(hyp):
    maps = [
        ("a. m.", "am"),
        ("p. m.", "pm"),
        ("u. s.", "us"),
        ("d. c.", "dc")
    ]
    for a, b in maps:
        hyp = hyp.replace(a, b)
    return hyp


def process_confusion(dataset, output, stop_words=None):
    pairs = []
    for data in tqdm(dataset):
        ref = data["ref"]
        hyp = data["text"]
        hyp = process_hyp(hyp)
        pairs.append((ref, (hyp, 1.0)))

    fd1, path1 = tempfile.mkstemp()
    fd2, path2 = tempfile.mkstemp()
    with open(path1, 'w') as fw1, open(path2, 'w') as fw2:
        for i, (trans, (text, _)) in enumerate(pairs):
            fw1.write('{} {}\n'.format(i, trans))
            fw2.write('{} {}\n'.format(i, text))

    p = subprocess.Popen(
        ['/home/cwhuang/kaldi/src/bin/align-text', f"ark:{path1}",
         f"ark:{path2}", 'ark,t:-'],
        stderr=None, stdout=subprocess.PIPE,
        universal_newlines=True
    )
    ret = p.communicate()[0]

    data = []
    conf_count = dict()
    for line in tqdm(ret.split('\n')):
        if len(line.strip()) == 0:
            continue
        i, confusion = line.strip().split(maxsplit=1)
        i = int(i)
        trans, (text, score) = pairs[i]

        for conf in confusion.strip().split(";"):
            a, b = conf.strip().split()
            if a == b or "<eps>" in {a, b} or a.isdigit() or b.isdigit():
                continue
            if stop_words is not None and a in stop_words or b in stop_words:
                continue
            conf = conf.strip()
            if conf not in conf_count:
                conf_count[conf] = 0
            conf_count[conf] += 1

        if len(confusion.strip()) > 0:
            data.append({
                "transcription": trans,
                "hypothesis": text,
                "confusion": confusion,
                "score": score
            })

    conf_sort = sorted(conf_count.items(), key=lambda x: x[1], reverse=True)
    for i, (conf, count) in enumerate(conf_sort):
        if i == 40:
            break
        print(conf, count)

    os.remove(path1)
    os.remove(path2)
    os.close(fd1)
    os.close(fd2)

    with open(output, 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["transcription", "hypothesis", "confusion", "score"])
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    args = parse_args()
    dataset = read_csv(args.file)
    stop_words = read_stop_words()
    process_confusion(dataset, args.output, stop_words)
