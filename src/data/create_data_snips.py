import argparse
import csv
import json
import os
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="dataset.json file")
    parser.add_argument("metadata", type=str, help="metadata.json file")
    parser.add_argument("output", type=str)
    args = parser.parse_args()
    return args


def process_text(text):
    if text.endswith('.'):
        text = text[:-1]
    text = text.replace('\n', '')
    return text.lower()


def get_metadata(metadata):
    data = json.load(open(metadata))
    text_map = {process_text(d["text"]): key for key, d in data.items()}
    return text_map


def get_utterance_text(utt):
    utterance = ""
    for text in utt["data"]:
        utterance += text['text']

    return process_text(utterance)


def make_dataset(dataset, metadata, output):
    data = []
    for intent, utts in dataset["intents"].items():
        for utt in utts["utterances"]:
            text = get_utterance_text(utt)
            try:
                id = metadata[text]
            except KeyError:
                print(f"'{text}' not in metadata")
                continue
            data.append({
                "id": id,
                "text": text,
                "label": intent
            })

    with open(output, 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["id", "text", "label"])
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    args = get_args()
    metadata = get_metadata(args.metadata)
    dataset = json.load(open(args.dataset))
    make_dataset(dataset, metadata, args.output)
