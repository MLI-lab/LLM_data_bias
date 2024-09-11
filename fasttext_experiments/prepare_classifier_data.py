import glob, os
import json
import sys
import re
import hashlib
import gzip
import os
import argparse
import jsonlines
import random
from tqdm import tqdm
import multiprocessing
import numpy as np


def get_jsonl_data(paths):
    lines = []
    paths = [paths]
    for file_path in paths:
        with open(file_path, "r") as f:
            with jsonlines.Reader(f) as jsonl_reader:
                for item in jsonl_reader:
                    #if random.random() < 0.02: # only take 2% of the data
                    lines.append(item["text"])
                        # lines.append(json.dumps(item))

    return lines


def get_args():
    parser = argparse.ArgumentParser(description="Extract text data to train classifiers")

    parser.add_argument("--reference", help="Input file path for reference data.")
    parser.add_argument("--unlabeled", help="Input file path for commoncrawl unlabeled data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random subset of data")
    parser.add_argument("--num_lines", type=int, default=10_000, help="Number of datapoints from each class")
    parser.add_argument("--output_file", help="Output data file path")

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


def main():
    """
    Command: python embeddings/prepare_classifier_data.py --reference ../../data/untokenized/redpajama/wiki_en/ --unlabeled ../../data/untokenized/common_crawl_v3_pre2023_0.01_frac_sample_jsonls_random/ --output_file ../../data/classification_data/wiki_common_crawl_20k_seed=42.csv
    """
    args = get_args()

    reference_path = args.reference
    unlabeled_path = args.unlabeled
    num_lines = args.num_lines
    seed = args.seed
    output_file = args.output_file

    random.seed(seed)

    reference_files = glob.glob(f"{reference_path}/*.jsonl")
    unlabeled_files = glob.glob(f"{unlabeled_path}/*.jsonl")

    random.shuffle(reference_files)
    random.shuffle(unlabeled_files)

    print("Collected filenames.")

    print("Reading reference data ...")

    with multiprocessing.Pool(32) as pool:
        reference_data_list = list(tqdm(pool.imap(get_jsonl_data, reference_files)))

    # reference_data = list(np.concatenate(reference_data, axis=0))

    count = 100_000
    with open(output_file, "w") as f:
        i = 0
        for reference_data in reference_data_list:
            for line in reference_data:
                f.write("__label__positive " + " ".join(line.splitlines()) + "\n")
                i+= 1
                if i >=count:
                    break
            if i >= count: 
                break
    # reference_data = get_jsonl_data(reference_files)

    print("Reading unlabeled data ...")
    with multiprocessing.Pool(32) as pool:
        unlabeled_data_list = list(tqdm(pool.imap(get_jsonl_data, unlabeled_files)))

    # unlabeled_data = list(np.concatenate(unlabeled_data, axis=0))


    # print(f"Total lengths: reference={len(reference_data)} and unlabeled={len(unlabeled_data)}")

    # unlabeled_data = unlabeled_data[:num_lines]

    with open(output_file, "a") as f:
        i = 0
        for unlabeled_data in unlabeled_data_list: 
            for line in unlabeled_data:
                f.write("__label__negative " + " ".join(line.splitlines()) + "\n")
                i+= 1
                if i >= count: 
                    break
            if i >= count: 
                break

if __name__ == "__main__":
    main()
