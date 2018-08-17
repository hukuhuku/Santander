#This code written by amaotone
#https://amalog.hateblo.jp/entry/kaggle-feature-management

import re
import time
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np

import argparse
import inspect

INPUT_DIR = './input/'
OUTPUT_DIR = './output/'
DATA_DIR = './data/'

def get_arguments(description = None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'./data/{self.name}_train.ftr'
        self.test_path = Path(self.dir) / f'./data/{self.name}_test.ftr'
    
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))


def get_data(feats=None,converting=False):
    if converting:
        dfs = [pd.read_feather(f'./data/{f}_train.ftr') for f in feats]
        dfs.append(pd.read_csv("./input/train.csv")[["ID","target"]])
        X_train = pd.concat(dfs, axis=1)
        dfs = [pd.read_feather(f'./data/{f}_test.ftr') for f in feats]
        dfs.append(pd.read_csv("./input/test.csv")["ID"])
        X_test = pd.concat(dfs, axis=1)
    else:
        X_train = pd.read_csv("./input/train.csv")
        X_test = pd.read_csv("./input/test.csv")
    return X_train, X_test


def get_leak_indexes():
    leak_indexes = np.load('./data/test_leak_indexes.npy')
    non_leak_indexes = np.load('./data/test_non_leak_indexes.npy')

    return leak_indexes,non_leak_indexes
    
def get_ugly_indexes():
    ugly_indexes = np.load('./data/test_ugly_indexes.npy')
    non_ugly_indexes = np.load('./data/test_non_ugly_indexes.npy')

    return ugly_indexes,non_ugly_indexes

def get_leak_submission():
    sub_leak = pd.read_csv('./output/baseline_leak.csv')
    return sub_leak

def get_leak_columns():
    return ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', 
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', 
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
        '6619d81fc', '1db387535','fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
       ]

def get_extra_columns():
    pattern_1964666 = pd.read_csv(DATA_DIR+'pattern-found/pattern_1964666.66.csv')
    pattern_1166666 = pd.read_csv(DATA_DIR+'pattern-found/pattern_1166666.66.csv')
    pattern_812666 = pd.read_csv(DATA_DIR+'pattern-found/pattern_812666.66.csv')
    pattern_2002166 = pd.read_csv(DATA_DIR+'pattern-found/pattern_2002166.66.csv')
    pattern_3160000 = pd.read_csv(DATA_DIR+'pattern-found/pattern_3160000.csv')
    pattern_3255483 = pd.read_csv(DATA_DIR+'pattern-found/pattern_3255483.88.csv')

    pattern_1964666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_1166666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_812666.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_2002166.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_3160000.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)
    pattern_3255483.drop(['Unnamed: 0','value_count'],axis=1,inplace=True)

    pattern_1166666.rename(columns={'8.50E+43': '850027e38'},inplace=True)

    l=[]
    l.append(pattern_1964666.columns.values.tolist())
    l.append(pattern_1166666.columns.values.tolist())
    l.append(pattern_812666.columns.values.tolist())
    l.append(pattern_2002166.columns.values.tolist())
    l.append(pattern_3160000.columns.values.tolist())
    l.append(pattern_3255483.columns.values.tolist())
    l.append(['ced6a7e91','9df4daa99','83c3779bf','edc84139a','f1e0ada11','73687e512','aa164b93b','342e7eb03',
                'cd24eae8a','8f3740670','2b2a10857','a00adf70e','3a48a2cd2','a396ceeb9','9280f3d04','fec5eaf1a',
                '5b943716b','22ed6dba3','5547d6e11','e222309b0','5d3b81ef8','1184df5c2','2288333b4','f39074b55',
                'a8b721722','13ee58af1','fb387ea33','4da206d28','ea4046b8d','ef30f6be5','b85fa8b27','2155f5e16'
               ])

    return l
    