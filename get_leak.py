import pandas as pd
import numpy as np

from functions import * 
from base import *

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


class find_leak_lags(Leak):
    def find_leak(self):
        max_nlags = len(cols) - 2
        
        for df,self_df in [(train,self.train),(test,self.test)]:
            try:
                tmp = df[["ID", "target"] + cols]
            except:
                tmp = df[["ID"]+cols]

            tmp["compiled_leak"] = 0
            tmp["nonzero_mean"] = train[transact_cols].apply(
                lambda x: np.expm1(np.log1p(x[x!=0]).mean()), axis=1
            )
    
            leaky_cols = []

            for i in range(max_nlags):
                c = "leaked_target_"+str(i)
                print("Processing lag",i)

                tmp[c] = get_leak(tmp,cols,i)
                leaky_cols.append(c)

                tmp = train.join(
                    tmp.set_index("ID")[leaky_cols+["compiled_leak", "nonzero_mean"]], 
                    on="ID", how="left"
                )[["ID", "target"] + cols + leaky_cols+["compiled_leak", "nonzero_mean"]]
        
                zeroleak = tmp["compiled_leak"]==0
                tmp.loc[zeroleak, "compiled_leak"] = tmp.loc[zeroleak, c]
            print("Leak values found in train",sum(tmp["compiled_leak"] > 0))
            
            self_df = tmp

if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')

    transact_cols = [f for f in train.columns if f not in ["ID", "target"]]
    y = np.log1p(train["target"]).values
    
    find_leak_lags().run()