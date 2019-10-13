import argparse
import csv
import subprocess
import os
import tempfile
import math
import re

from tqdm import tqdm
from scipy.special import softmax

from create_data_confusion_csv import process_hyp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sausage", type=str, help="sausage file")
    parser.add_argument("nbest", type=str, help="nbest file")
    parser.add_argument("score", type=str, help="nbest score file")
    parser.add_argument("words", type=str, help="words.txt")
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    return args


def read_stop_words():
    ret = set()
    for line in open("data/stop_words.txt"):
        ret.add(line.strip())
    return ret


def read_words(filename):
    words = dict()
    for line in open(filename):
        word, wid = line.strip().split()
        words[int(wid)] = word
    return words


def read_nbest(filename):
    nbest = dict()
    for line in open(filename):
        uid, *words = line.strip().split()
        uid, n = uid.rsplit("-", maxsplit=1)
        if uid not in nbest:
            nbest[uid] = []
        nbest[uid].append(words)
    return nbest


def read_score(filename):
    scores = dict()
    for line in open(filename):
        uid, score = line.strip().split()
        score = float(score)
        uid, n = uid.rsplit("-", maxsplit=1)
        if uid not in scores:
            scores[uid] = []
        scores[uid].append(score)
    return scores


def process_word(word):
    return word[:-1].lower() if word.endswith(".") else word.lower()


def read_sausage(filename, words):
    sausages = []
    for line in open(filename):
        uid, sau = line.strip().split(maxsplit=1)
        saus = [x[1:-1].split() for x in re.findall("\[.*?\]", sau)]
        saus = [
            [
                (process_word(words[int(wid)]), float(score))
                for wid, score in zip(sau[::2], sau[1::2])
            ]
            for sau in saus
        ]
        sausages.append({
            "id": uid,
            "sausage": saus
        })
    return sausages


def create_confusion(sausages, nbests, scores, stop_words=None):
    data = []
    conf_count = dict()
    for sausage in sausages:
        uid = sausage["id"]
        saus = sausage["sausage"]
        sau_words = [[process_word(w) for w, s in sau] for sau in saus]
        sau_scores = [[s for w, s in sau] for sau in saus]
        nbest = nbests[uid]
        if len(nbest) <= 1:
            continue
        score = softmax(scores[uid][1:])
        top_hyp = nbest[0]
        top_hyp = [process_word(w) for w in top_hyp]
        for hyp, hyp_score in zip(nbest[1:], score):
            hyp = [process_word(w) for w in hyp]
            wtid = wid = 0
            sid = 0
            confs = []
            while wtid < len(top_hyp) and wid < len(hyp):
                wt_in = top_hyp[wtid] in sau_words[sid]
                w_in = hyp[wid] in sau_words[sid]
                if wt_in and not w_in:
                    confs.append(f"{top_hyp[wtid]} <eps>")
                    wtid += 1
                elif not wt_in and w_in:
                    confs.append(f"<eps> {hyp[wid]}")
                    wid += 1
                elif wt_in and w_in:
                    confs.append(f"{top_hyp[wtid]} {hyp[wid]}")
                    wtid += 1
                    wid += 1
                sid += 1

            for conf in confs:
                a, b = conf.strip().split()
                a, b = sorted((a, b))
                conf = " ".join((a, b))
                if a == b or "<eps>" in {a, b} or a.isdigit() or b.isdigit():
                    continue
                if stop_words is not None and a in stop_words or b in stop_words:
                    continue
                if conf not in conf_count:
                    conf_count[conf] = 0
                conf_count[conf] += 1

            data.append({
                "transcription": " ".join(top_hyp),
                "hypothesis": " ".join(hyp),
                "confusion": " ; ".join(confs),
                "score": hyp_score
            })

    conf_sort = sorted(conf_count.items(), key=lambda x: x[1], reverse=True)
    for i, (conf, count) in enumerate(conf_sort):
        if i == 40:
            break
        print(conf, count)

    return data


if __name__ == "__main__":
    args = parse_args()
    stop_words = read_stop_words()
    words = read_words(args.words)
    sausages = read_sausage(args.sausage, words)
    nbests = read_nbest(args.nbest)
    scores = read_score(args.score)
    confusions = create_confusion(sausages, nbests, scores, stop_words)

    with open(args.output, 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["transcription", "hypothesis", "confusion", "score"])
        writer.writeheader()
        writer.writerows(confusions)
