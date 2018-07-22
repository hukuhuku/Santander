from base import Feature, get_arguments, generate_features

import pandas as pd
import numpy as np 
from scipy.stats import skew

from sklearn import random_projection

class select_features(Feature):
    def create_features(self):
        select =  [
            'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
            '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
            '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
            'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
            '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
            '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
            '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
            'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
            ]
        self.train = train[select]
        self.test = test[select]

class statics(Feature):
    def create_features(self):
        
        train_zeros = pd.DataFrame({'Percent_zero': ((train.values) == 0).mean(axis=0),
                                'Column': train.columns})
    
        high_vol_columns = train_zeros['Column'][train_zeros['Percent_zero'] < 0.70].values
        low_vol_columns = train_zeros['Column'][train_zeros['Percent_zero'] >= 0.70].values
        
        #train=train.replaceにするとtrainがローカル変数になってしまう
        tmp_train = train.replace({0:np.nan})
        tmp_test = test.replace({0:np.nan})

        cluster_sets = {"low":low_vol_columns, "high":high_vol_columns}
        for cluster_key in cluster_sets:
            for df,self_df in [(tmp_train,self.train),(tmp_test,self.test)]:
                self_df["count_not0_"+cluster_key] = df[cluster_sets[cluster_key]].count(axis=1)
                self_df["sum_"+cluster_key] = df[cluster_sets[cluster_key]].sum(axis=1)
                self_df["var_"+cluster_key] = df[cluster_sets[cluster_key]].var(axis=1)
                self_df["median_"+cluster_key] = df[cluster_sets[cluster_key]].median(axis=1)
                self_df["mean_"+cluster_key] = df[cluster_sets[cluster_key]].mean(axis=1)
                self_df["std_"+cluster_key] = df[cluster_sets[cluster_key]].std(axis=1)
                self_df["max_"+cluster_key] = df[cluster_sets[cluster_key]].max(axis=1)
                self_df["min_"+cluster_key] = df[cluster_sets[cluster_key]].min(axis=1)
                self_df["skew_"+cluster_key] = df[cluster_sets[cluster_key]].skew(axis=1)
                self_df["kurtosis_"+cluster_key] = df[cluster_sets[cluster_key]].kurtosis(axis=1)

        del(tmp_train)
        del(tmp_test)

class RandomProjection(Feature):
    def create_features(self):
        n_com = 100
        transformer = random_projection.SparseRandomProjection(n_components = n_com)

        self.train = pd.DataFrame(transformer.fit_transform(train))
        self.test = pd.DataFrame(transformer.transform(test))
        
        columns = ["RandomProjection{}".format(i) for i in range(n_com)]
        self.train.columns = columns
        self.test.columns = columns


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('input/train.csv').drop(["ID","target"],axis=1)
    test = pd.read_csv('input/test.csv').drop("ID",axis=1)

    