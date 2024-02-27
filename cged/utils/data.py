import json
import os, pickle, sys, copy
# import pandas as pd
import random
import numpy as np
from typing import List
# import torch
from tqdm import tqdm


def handle_ch_character(s):
    # \u3000 = 空格
    return s.strip().replace(',', '，').replace("\u3000", '').replace("\u200b", ''). \
        replace("\xa0", '').replace("\ufeff", '')


class Data:
    def __init__(self, test: bool = False):
        self.train_loader = []
        self.valid_loader = []
        self.test = test

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Train  Instance Number: %s" % (len(self.train_loader)))
        print("     Valid  Instance Number: %s" % (len(self.valid_loader)))
        # print("     Test   Instance Number: %s" % (len(self.test_loader)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def generate_instance(self, args):
        print('generate instance...')
        self.train_loader = self.read_file(args.dataset_dir, 'train')
        self.valid_loader = self.read_file(args.dataset_dir, 'dev')

    @staticmethod
    def read_file(file_path: str = '', type=None):
        with open(os.path.join(file_path, type + '.json')) as f:
            lines = json.loads(f.read())
            f.close()

        return lines


def build_data(args, test: bool = False):
    """处理数据"""
    if args.cache_file:
        file = args.cache_file
    else:
        file = args.cache_data_directory + args.dataset_name + "_data.pkl"

    if os.path.exists(file) and not args.refresh:
        # from cache
        print('data is from cache: {}'.format(file))
        data = load_data_setting(args)
    else:
        data = Data(test)
        data.generate_instance(args)
        save_data_setting(data, args, file)
    return data


def save_data_setting(data, args, file):
    new_data = copy.deepcopy(data)
    data.show_data_summary()
    if not os.path.exists(args.cache_data_directory):
        os.makedirs(args.cache_data_directory)

    with open(file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting is saved to file: ", file)

def load_data_setting(args):
    if args.cache_file:
        saved_path = args.cache_file
    else:
        saved_path = args.cache_data_directory + args.dataset_name + "_data.pkl"

    with open(saved_path, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting is loaded from file: ", saved_path)

    _avg_length = 0
    for valid in data.valid_loader:
        _avg_length += len(valid['src'])
    print(_avg_length / len(data.valid_loader))
    #
    # base = 64
    # data.train_loader = data.train_loader[:base * 4]
    # data.valid_loader = data.valid_loader[:base * 2]

    data.show_data_summary()
    return data


if __name__ == '__main__':
    from transformers import BertTokenizer
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from cged.config.base import get_args

    args, _ = get_args()
    tokenizer = BertTokenizer.from_pretrained(args.lm)
    data = build_data(args)
