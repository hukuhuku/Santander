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


train = pd.read_csv(INPUT_DIR+"train.csv")
test = pd.read_csv(INPUT_DIR+"test.csv")

transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
y = np.log1p(train["target"]).values

cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', 
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', 
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
        '6619d81fc', '1db387535', 

        'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
        ]

extra_cols= [['ced6a7e91','9df4daa99','83c3779bf','edc84139a','f1e0ada11','73687e512','aa164b93b','342e7eb03',
        'cd24eae8a','8f3740670','2b2a10857','a00adf70e','3a48a2cd2','a396ceeb9','9280f3d04','fec5eaf1a',
        '5b943716b','22ed6dba3','5547d6e11','e222309b0','5d3b81ef8','1184df5c2','2288333b4','f39074b55',
        'a8b721722','13ee58af1','fb387ea33','4da206d28','ea4046b8d','ef30f6be5','b85fa8b27','2155f5e16']
]

def get_beautiful_test(test):
    test_rnd = np.round(test.iloc[:, 1:], 2)
    ugly_indexes = []
    non_ugly_indexes = []
    for idx in tqdm(range(len(test))):
        if not np.all(
            test_rnd.iloc[idx, :].values==test.iloc[idx, 1:].values
        ):
            ugly_indexes.append(idx)
        else:
            non_ugly_indexes.append(idx)
    print(len(ugly_indexes), len(non_ugly_indexes))
    np.save('test_ugly_indexes', np.array(ugly_indexes))
    np.save('test_non_ugly_indexes', np.array(non_ugly_indexes))
    #test = test.iloc[non_ugly_indexes].reset_index(drop=True)
    return test, non_ugly_indexes, ugly_indexes


def fast_get_leak(df, cols,extra_feats, lag=0):
    f1 = cols[:-lag-2]
    f2 = cols[lag+2:]
    for ef in extra_feats:
        f1 += ef[:lag-2]
        f2 += ef[lag+2:]
        
    d1 = df[f1].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = df[f2].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = df[cols[lag]]
    #d2 = d2[d2.pred != 0] ### to make output consistent with Hasan's function
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    d4 = d1[~d1.duplicated(['key'], keep=False)]

    print("del by duplicated_cols{} => {}".format(d1.shape[0],d4.shape[0]))
    print("del by duplicated_cols{} => {}".format(d2.shape[0],d3.shape[0]))
    #d5 = d4.merge(d3, how='inner', on='key')#重複は結構多いのでどれかが当たるのを期待する、時系列columnsを探してextra_colsに追加する

    d = d1.merge(d3, how='left', on='key')
    gc.collect()
    return d.pred.fillna(0)

def compiled_leak_result(use_train=True):
    use_cols = ["ID"] + cols
    if use_train == True:
        df = train
        use_cols = ["ID","target"] + cols
    else:
        df = test
      
    for ef in extra_cols:
        use_cols += ef

    max_nlags = len(cols) - 2
    df_leak = df[use_cols]

    df_leak["compiled_leak"] = 0
    df_leak["nonzero_mean"] = df[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )
    
    scores = []
    leaky_value_counts = []
    leaky_value_corrects = []
    leaky_cols = []
    
    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        
        print('Processing lag', i)
        df_leak[c] = fast_get_leak(df_leak, cols,extra_cols,i)
        
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
            result = dict(
                score=scores, 
                leaky_count=leaky_value_counts,
                leaky_correct=leaky_value_corrects,
            )
        else:
            result = dict(
                leaky_count = leaky_value_counts
            )
    
    return df_leak, result

def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

def main():
    #test, non_ugly_indexes, ugly_indexes = get_beautiful_test(test)
    ugly_indexes = np.load('test_ugly_indexes.npy')
    non_ugly_indexes = np.load("test_non_ugly_indexes.npy")
    test["target"] = train["target"].mean()

    train_leak, result = compiled_leak_result()
    result = pd.DataFrame.from_dict(result, orient='columns')
    result.to_csv(DATA_DIR+'train_leaky_stat.csv', index=False)
    best_score = np.min(result['score'])
    best_lag = np.argmin(result['score'])
    print('best_score', best_score, '\nbest_lag', best_lag)
 
    leaky_cols = [c for c in train_leak.columns if 'leaked_target_' in c]
    train_leak = rewrite_compiled_leak(train_leak, best_lag)
    train_res = train_leak[leaky_cols+['compiled_leak']].replace(0.0, np.nan)
    train_res.to_csv(DATA_DIR+'train_leak.csv', index=False)
    gc.collect()

    test_leak, test_result = compiled_leak_result(use_train = False)
    test_result = pd.DataFrame.from_dict(test_result, orient='columns')
    test_result.to_csv(DATA_DIR+'test_leaky_stat.csv', index=False)
    test_leak = rewrite_compiled_leak(test_leak, best_lag)
    test_res = test_leak[leaky_cols+['compiled_leak']].replace(0.0, np.nan)
    test_res.to_csv(DATA_DIR+'test_leak.csv', index=False)
    test_leak.loc[test_leak["compiled_leak"]==0, "compiled_leak"] = test_leak.loc[test_leak["compiled_leak"]==0, "nonzero_mean"]
    gc.collect()

    sub = pd.read_csv(INPUT_DIR+"test.csv", usecols=["ID"])
    sub["target"] = 0
    sub.loc[non_ugly_indexes, "target"] = test_leak["compiled_leak"].values
    sub.to_csv(OUTPUT_DIR+f"non_fake_sub_lag_{best_lag}.csv", index=False)
    print(f"non_fake_sub_lag_{best_lag}.csv saved")
    
if __name__ == "__main__":
    main()