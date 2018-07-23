import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from math import sqrt

import dask.dataframe as dd
from dask.multiprocessing import get

from tqdm import tqdm

def get_beautiful_test(test):
    #小数点以下2桁以上とそれ以外でpublicとprivateに分かれたりする
    #それを分離
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
    test = test.iloc[non_ugly_indexes].reset_index(drop=True)
    return test, non_ugly_indexes, ugly_indexes

def get_leak(df, cols, lag=0):
    #lagの分だけずらしてtuple型にする、一致だったら時系列的にあってると解釈できる
    d1 = df[cols[:-lag-2]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = df[cols[lag+2:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = df[cols[lag]]
    #d2 = d2[d2.pred != 0] ### to make output consistent with Hasan's function
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return d1.merge(d3, how='left', on='key').pred.fillna(0)

def compiled_leak_result():
    
    max_nlags = len(cols) - 2
    train_leak = train[["ID", "target"] + cols]
    train_leak["compiled_leak"] = 0
    train_leak["nonzero_mean"] = train[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )
    
    scores = []
    leaky_value_counts = []
    leaky_value_corrects = []
    leaky_cols = []
    
    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        
        print('Processing lag', i)
        train_leak[c] = _get_leak(train_leak, cols, i)
        
        leaky_cols.append(c)
        train_leak = train.join(
            train_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]], 
            on="ID", how="left"
        )[["ID", "target"] + cols + leaky_cols+["compiled_leak", "nonzero_mean"]]
        
        zeroleak = train_leak["compiled_leak"]==0
        train_leak.loc[zeroleak, "compiled_leak"] = train_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(train_leak["compiled_leak"] > 0))
        _correct_counts = sum(train_leak["compiled_leak"]==train_leak["target"])
        leaky_value_corrects.append(_correct_counts/leaky_value_counts[-1])
        print("Leak values found in train", leaky_value_counts[-1])
        
        print(
            "% of correct leaks values in train ", 
            leaky_value_corrects[-1]
        )
        tmp = train_leak.copy()
        tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        print(
            'Score (filled with nonzero mean)', 
            scores[-1]
        )
    result = dict(
        score=scores, 
        leaky_count=leaky_value_counts,
        leaky_correct=leaky_value_corrects,
    )
    return train_leak, result

def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

def fast_get_leak(df, cols, lag=0):
    d1 = df[cols[:-lag-2]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = df[cols[lag+2:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = df[cols[lag]]
    #d2 = d2[d2.pred != 0] ### to make output consistent with Hasan's function
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return d1.merge(d3, how='left', on='key').pred.fillna(0)


def compiled_leak_result_test(max_nlags):
    test_leak = test[["ID", "target"] + cols]
    test_leak["compiled_leak"] = 0
    test_leak["nonzero_mean"] = test[transact_cols].apply(
        lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
    )
    
    scores = []
    leaky_value_counts = []
    # leaky_value_corrects = []
    leaky_cols = []
    
    for i in range(max_nlags):
        c = "leaked_target_"+str(i)
        
        print('Processing lag', i)
        test_leak[c] = fast_get_leak(test_leak, cols, i)
        
        leaky_cols.append(c)
        test_leak = test.join(
            test_leak.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]], 
            on="ID", how="left"
        )[["ID", "target"] + cols + leaky_cols+["compiled_leak", "nonzero_mean"]]
        zeroleak = test_leak["compiled_leak"]==0
        test_leak.loc[zeroleak, "compiled_leak"] = test_leak.loc[zeroleak, c]
        leaky_value_counts.append(sum(test_leak["compiled_leak"] > 0))
        #_correct_counts = sum(train_leak["compiled_leak"]==train_leak["target"])
        #leaky_value_corrects.append(_correct_counts/leaky_value_counts[-1])
        print("Leak values found in test", leaky_value_counts[-1])
        #print(
        #    "% of correct leaks values in train ", 
        #    leaky_value_corrects[-1]
        #)
        #tmp = train_leak.copy()
        #tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, "nonzero_mean"]
        #scres.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
        #print(
        #    'Score (filled with nonzero mean)', 
        #    scores[-1]
        #)
    result = dict(
        # score=scores, 
        leaky_count=leaky_value_counts,
        # leaky_correct=leaky_value_corrects,
    )
    return test_leak, result
