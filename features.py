from base import Feature, get_arguments, generate_features

import pandas as pd
import numpy as np 
from scipy.stats import skew

from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection._split import check_cv
from sklearn.decomposition import PCA

def replace_columns(train,test,class_name,cols_len):
    columns = ["{}{}".format(class_name,i) for i in range(cols_len)]
    train.columns = columns
    test.colummns = columns

    return train,test

def get_timecolumns():
    return ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', 
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', 
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
        '6619d81fc', '1db387535','fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
       ]

class select_features(Feature):
    def create_features(self):
        select = get_timecolumns()
        self.train = train[select]
        self.test = test[select]

class raw_data(Feature):
    def create_features(self):
        self.train = train
        self.test = test

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


class diff(Feature):
    #データが時系列なので変化の大きさなどを取得する
    def create_features(self):
        cols = get_timecolumns()
        
        self.train = pd.DataFrame(np.diff(train[cols]))
        self.test  = pd.DataFrame(np.diff(test[cols]))
       
        columns = ["diff{}".format(i) for i in range(len(cols)-1)]
  
        self.train.columns = columns
        self.test.columns = columns
    
class timespan(Feature):
    def create_features(self):
        cols = get_timecolumns()

        for time in [3,10,20,30,len(cols)]:
            for df,self_df in [(train,self.train),(test,self.test)]:
                tmp_df = df[cols[:time]].replace({0:np.nan})
                self_df["mean_{}".format(time)] = tmp_df.mean(axis=1)
                self_df["sum_{}".format(time)]  = tmp_df.sum(axis=1)
                self_df["diff_mean_{}".format(time)] = tmp_df.diff(axis=1).mean(axis=1)
                self_df["min_{}".format(time)] = tmp_df.min(axis=1)
                self_df["max_{}".format(time)] = tmp_df.max(axis=1)
                self_df["count_not_0_{}".format(time)] = tmp_df.count(axis=1)
 

class RandomProjection(Feature):
    def create_features(self):
        n_com = 100
        transformer = random_projection.SparseRandomProjection(n_components = n_com)

        self.train = pd.DataFrame(transformer.fit_transform(train))
        self.test = pd.DataFrame(transformer.transform(test))
        
        columns = ["RandomProjection{}".format(i) for i in range(n_com)]
        self.train.columns = columns
        self.test.columns = columns


class Principal_Component_Analysis(Feature):
    def create_features(self):
        n_com = 100

        pca = PCA()

        pca.fit(train)
        self.train = pd.DataFrame(pca.transform(train))
        self.test = pd.DataFrame(pca.transform(test))

        t = pd.DataFrame(pca.explained_variance_ratio_)

        #寄与度が50パーセントになるように
        for i in range(1,len(t)):
            t.loc[i] += t.loc[i-1]
            if t.loc[i].values > 0.5:#need tune
                end = i 
                break
        del(t)

        self.train = self.train[self.train.columns[:end]]
        self.test = self.test[self.test.columns[:end]]

    
        columns = ["PCA{}".format(i) for i in range(end)]
        self.train.columns = columns
        self.test.columns = columns
        

class tSVD(Feature):
    def create_features(self):
        n_com = 100

        transformer =TruncatedSVD(n_components = n_com)
        
        self.train = pd.DataFrame(transformer.fit_transform(train))
        self.test = pd.DataFrame(transformer.transform(test))
        
        columns = ["TruncatedSVD{}".format(i) for i in range(n_com)]
        self.train.columns = columns
        self.test.columns = columns


if __name__ == '__main__':
    args = get_arguments()

    train = pd.read_csv('input/train.csv').drop(["ID","target"],axis=1)
    test = pd.read_csv('input/test.csv').drop("ID",axis=1)

    generate_features(globals(), args.force)

