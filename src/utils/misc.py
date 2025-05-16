#!/usr/bin/python3
# -*- coding: utf-8 -*-

from multiprocessing import Pool, TimeoutError
from tqdm import tqdm
from functools import partial
import json, jsonlines
from loguru import logger
import pickle
import torch
import numpy as np
import random
from transformers import set_seed


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def save_json(file, data_list):
    logger.info(f"Saving to {file}")
    with open(file, "w") as f:
        json.dump(data_list, f, indent=4)

def load_json(file, show_dir=True):
    if show_dir:
        logger.info(f"Loading from {file}")
    with open(file) as f:
        data = json.load(f)
    return data

def save_jsonl(file, data_list):
    logger.info(f"Saving to {file}")
    with jsonlines.open(file, "w") as f:
        for data in data_list:
            f.write(data)

def load_jsonl(file, show_dir=True):
    if show_dir:
        logger.info(f"Loading from {file}")
    data_list = []
    with jsonlines.open(file) as f:
        for data in f:
            data_list.append(data)
    return data_list

def save_pickle(file, data):
    logger.info(f"Saving to {file}")
    with open(file, "wb") as f:
        pickle.dump(data, f)

def load_pickle(file):
    logger.info(f"Loading from {file}")
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def levenshtein_distance_word_level(s1, s2):
    # edit distance between two sentences
    words1, words2 = s1.split(), s2.split()
    m, n = len(words1), len(words2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]

