from base import Feature, get_arguments, generate_features

import pandas as pd
import numpy as np 
from scipy.stats import skew

from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection._split import check_cv
from sklearn.decomposition import PCA

from base import *

def replace_columns(train,test,class_name,cols_len):
    columns = ["{}{}".format(class_name,i) for i in range(cols_len)]
    train.columns = columns
    test.colummns = columns

    return train,test

class select_features(Feature):
    def create_features(self):
        select = get_leak_columns()
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


    
class timespan(Feature):
    def create_features(self):
        cols = get_leak_columns()
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

    train_leak = pd.read_csv("data/train_leak.csv")
    test_leak = pd.read_csv("data/test_leak.csv")
    del(test_leak["Unnamed: 0"])

    train = train[train_leak["compiled_leak"] == 0]
    test = test[test_leak["compiled_leak"] == 0]
    
    generate_features(globals(), args.force)

