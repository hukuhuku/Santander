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


class Leak(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'./data/{self.name}_train.tft'
        self.test_path = Path(self.dir) / f'./data/{self.name}_test.ftr'
    
    def run(self):
        with timer(self.name):
            self.find_leak()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def find_leak(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))

def get_data(feats=None):
    dfs = [pd.read_feather(f'./data/{f}_train.ftr') for f in feats]
    dfs.append(pd.read_csv("./input/train.csv")[["ID","target"]])
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'./data/{f}_test.ftr') for f in feats]
    dfs.append(pd.read_csv("./input/test.csv")["ID"])
    X_test = pd.concat(dfs, axis=1)

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