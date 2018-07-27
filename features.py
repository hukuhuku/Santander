from base import Feature, get_arguments, generate_features

import pandas as pd
import numpy as np 
from scipy.stats import skew

from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection._split import check_cv

def get_timecolumns():
    return ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
        'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', 
        '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 
        'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', 
        '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
        '6619d81fc', '1db387535', 
        'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
       ]

class select_features(Feature):
    def create_features(self):
        select = get_timecolumns()
        self.train = train[select]
        self.test = test[select]
"""
＃整理中
class RandomForestClassfier(Feature):
    #https://www.kaggle.com/sggpls/pipeline-kernel-xgb-fe-lb1-39
    def __init__(self):
        Feature.__init__(self)
        self.estimator = estimator
        self.n_classes = n_classes
        self.cv = cv

    def get_rfc():
        return RandomForestClassifier(
        n_estimators=100,
        max_features=0.5,
        max_depth=None,
        max_leaf_nodes=270,
        min_impurity_decrease=0.0001,
        random_state=123,
        n_jobs=-1
    )

    def _get_labels(self, y):
        y_labels = np.zeros(len(y))
        y_us = np.sort(np.unique(y))
        step = int(len(y_us) / self.n_classes)
        
        for i_class in range(self.n_classes):
            if i_class + 1 == self.n_classes:
                y_labels[y >= y_us[i_class * step]] = i_class
            else:
                y_labels[
                    np.logical_and(
                        y >= y_us[i_class * step],
                        y < y_us[(i_class + 1) * step]
                    )
                ] = i_class
        return y_labels



    def create_features(self):
        y = train["target"]
        y_labels = self._get_labels(y)
        cv = check_cv(self.cv, y_labels, classifier=is_classifier(self.estimator))
        self.estimators_ = []
        
        for train_index, _ in cv.split(train, y_labels):
            self.estimators_.append(
                clone(self.estimator).fit(train[train_index], y_labels[train_index])
            )

        X_prob = np.zeros((test.shape[0], self.n_classes))
        X_pred = np.zeros(test.shape[0])
        
        for estimator, (_, test) in zip(self.estimators_, cv.split(X)):
            X_prob[test] = estimator.predict_proba(X[test])
            X_pred[test] = estimator.predict(X[test])
        return np.hstack([X_prob, np.array([X_pred]).T])


"""
        
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
        print("OK")
        cols = get_timecolumns()
        
        self.train = pd.DataFrame(np.diff(train[cols]))
        self.test  = pd.DataFrame(np.diff(test[cols]))
       
        columns = ["diff{}".format(i) for i in range(len(cols)-1)]
  
        self.train.columns = columns
        self.test.columns = columns
    
        
class RandomProjection(Feature):
    def create_features(self):
        n_com = 100
        transformer = random_projection.SparseRandomProjection(n_components = n_com)

        self.train = pd.DataFrame(transformer.fit_transform(train))
        self.test = pd.DataFrame(transformer.transform(test))
        
        columns = ["RandomProjection{}".format(i) for i in range(n_com)]
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

