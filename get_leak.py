import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


import lightgbm as lgb
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import mode, skew, kurtosis, entropy
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

from multiprocessing import Pool
CPU_CORES = 1

import dask.dataframe as dd
from dask.multiprocessing import get

from tqdm import tqdm
from base import *

import gc
gc.collect();

INPUT_DIR = './input/'
OUTPUT_DIR = './output/'
DATA_DIR = './data/'

train,test = get_data()
transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
y = np.log1p(train["target"]).values

cols = get_leak_columns()
extra_cols= get_extra_columns()
max_nlags = len(cols) - 2

use_cols = ["ID","target"] + cols
for ef in extra_cols:
    use_cols += ef

def fast_get_leak(df, cols,extra_feats, lag=0):
    f1 = cols[:-lag-2]
    f2 = cols[lag+2:]

    for ef in extra_feats:
        f1 += ef[:-lag-2]
        f2 += ef[lag+2:]
        
    d1 = df[f1].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = df[f2].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = df[cols[lag]]
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]
    d5 = d4.merge(d3, how='inner', on='key')
    
    d6 = d1.merge(d5, how='left', on='key')
    d6.to_csv('extra_d6.csv')
        
    print("del by duplicated_cols{} => {}".format(d1.shape[0],d4.shape[0]))
    print("del by duplicated_cols{} => {}".format(d2.shape[0],d3.shape[0]))
    
    gc.collect()
    return d1.merge(d5, how='left', on='key').pred.fillna(0)

def compiled_leak_result(use_train=True,best_lag=None):
    if use_train:
        df = train
    else:
        df = test
        use_cols.remove("target")

    df_leak = df[use_cols]
    df_leak["compiled_leak"] = 0
    df_leak["nonzero_mean"] = df[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )

    leaky_value_counts = []
    leaky_value_corrects = []
    leaky_cols = []
    scores = []
    
    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        
        print('Processing lag', i)
        df_leak[c] = fast_get_leak(df_leak,cols,extra_cols,i)
        
        leaky_cols.append(c)
        df_leak = df.join(
            df_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]], 
            on="ID", how="left"
        )
        zeroleak = df_leak["compiled_leak"]==0
        df_leak.loc[zeroleak, "compiled_leak"] = df_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(df_leak["compiled_leak"] > 0))
    
        print("{} of count leaks values".format(leaky_value_counts[-1]))

        if use_train:
            _correct_counts = sum(df_leak["compiled_leak"]==df_leak["target"])
            leaky_value_corrects.append(_correct_counts)
            print("{} of correct_leaks values".format(leaky_value_corrects[-1]))
            tmp = df_leak.copy()
            tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
            scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
            print('Score (filled with nonzero mean)', scores[-1])
    if use_train:
        result = dict(
            score=scores, 
            leaky_count=leaky_value_counts,
            leaky_correct=leaky_value_corrects,
        )
    else:
        result = dict(leaky_count = leaky_value_counts)
    
    return df_leak, result

def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

def main():
    
    train_leak,train_result = compiled_leak_result(use_train=True)
    best_lag = np.argmin(train_result['score'])
    print("best_lag",best_lag)
    
    test_leak,test_result = compiled_leak_result(use_train=False)

    test_leak = rewrite_compiled_leak(test_leak,lag=best_lag)
    test_leak.to_csv("test_leak.csv",index=True)
    test_leak.loc[test_leak["compiled_leak"]==0, "compiled_leak"] = test_leak.loc[test_leak["compiled_leak"]==0, "nonzero_mean"]
    gc.collect()
    
    sub = pd.read_csv(INPUT_DIR+"test.csv", usecols=["ID"])
    sub["target"] =  test_leak["compiled_leak"].values
    sub.to_csv(OUTPUT_DIR+"non_fake_sub_lag_{}.csv".format(best_lag), index=False)

if __name__ == "__main__":
    main()