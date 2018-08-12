#leakを探すために書いた関数などを保存する
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import skew
import time
import gc
import tqdm
from base import *

def get_leak(df,cols,lag=0):#write myself
    d1 = df[cols[2+lag:]].apply(tuple,axis=1).to_frame().rename(columns={0: 'key'})
    d2 = df[cols[:-2-lag]].apply(tuple,axis=1).to_frame().rename(columns={0: 'key'})
    d2["pred"] = df[cols[lag]]
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return pd.merge(d1,d3,how="left",on="key")


def search_leak(df,rows,cols):
    Flag = True
    tuple_df = df.loc[:,cols[1:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})

    while Flag:
        tmp = tuple(df.loc[rows[-1],cols[:-1]])
        try:
            rows.append(tuple_df[tuple_df["key"] == tmp].index[0])
        except:
            Flag = False