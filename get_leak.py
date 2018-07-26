import pandas as pd
import numpy as np

from functions import * 
from base import *

from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', 
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', 
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
        '6619d81fc', '1db387535', 
        'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
       ]

def get_leak(df, cols, lag=0):
    #lagの分だけずらしてtuple型にする、一致だったら時系列的にあってると解釈できる
    d1 = df[cols[:-lag-2]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2 = df[cols[lag+2:]].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
    d2['pred'] = df[cols[lag]]
    #d2 = d2[d2.pred != 0] ### to make output consistent with Hasan's function
    d3 = d2[~d2.duplicated(['key'], keep=False)]
    return d1.merge(d3, how='left', on='key').pred.fillna(0)


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

def rewrite_compiled_leak(leak_df, lag):
    leak_df["compiled_leak"] = 0
    for i in range(lag):
        c = "leaked_target_"+str(i)
        zeroleak = leak_df["compiled_leak"]==0
        leak_df.loc[zeroleak, "compiled_leak"] = leak_df.loc[zeroleak, c]
    return leak_df

class find_leak_lags(Leak):
    def find_leak(self):
        max_nlags = len(cols) - 2
        
        for df in [train,test]:
            try:
                tmp = df[["ID", "target"] + cols]
                scores = []
            except:
                tmp_train = tmp
                tmp = df[["ID"]+cols]

            tmp["compiled_leak"] = 0
            tmp["nonzero_mean"] = df[transact_cols].apply(
                lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
            )
    
            leaky_cols = []
            

            for i in tqdm(range(max_nlags)):
                c = "leaked_target_"+str(i)

                tmp[c] = get_leak(df,cols,i)
                leaky_cols.append(c)

                zeroleak = tmp["compiled_leak"]==0
                tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, c]
                
                try:
                    scores.append(np.sqrt(mean_squared_error(y, np.log1p(tmp["compiled_leak"]).fillna(14.49))))
                except:
                    pass

            print("Leak values found ",sum(tmp["compiled_leak"] > 0))

            try:
                best_score = np.min(scores)
                best_lag = np.argmin(scores)
                print('best_score', best_score, '\nbest_lag', best_lag)
                del(scores)
            except:
                best_lag = 29
                tmp = rewrite_compiled_leak(tmp, best_lag)
                
                leak_index = tmp[tmp["compiled_leak"] >0].index
                not_leak_index = tmp[tmp["compiled_leak"] == np.nan].index
                np.save('test_leak_indeies', np.array(leak_index))
                np.save('test_not_leak_indeies', np.array(not_leak_index))


        self.train = tmp_train
        self.test = tmp

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')

    #get_beautiful_test(test)
    transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
    y = np.log1p(train["target"]).values
    
    find_leak_lags().run().save()