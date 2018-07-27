#LB1.37
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import skew
import time
import gc
import tqdm
from base import *

gc.enable() 


def fit_predict(data, y, test):
    folds = KFold(n_splits=5, shuffle=True, random_state=1)
    # Convert to lightgbm Dataset
    dtrain = lgb.Dataset(data=data, label=np.log1p(y['target']), free_raw_data=False)
    # Construct dataset so that we can use slice()
    dtrain.construct()
    # Init predictions
    sub_preds = np.zeros(test.shape[0])
    oof_preds = np.zeros(data.shape[0])
    # Lightgbm parameters
    # Optimized version scores 0.40
    # Step |   Time |      Score |      Stdev |   p1_leaf |   p2_subsamp |   p3_colsamp |   p4_gain |   p5_alph |   p6_lamb |   p7_weight |
    #   41 | 00m04s |   -1.36098 |    0.02917 |    9.2508 |       0.7554 |       0.7995 |   -3.3108 |   -0.1635 |   -0.9460 |      0.6485 |
    lgb_params = {
        'objective': 'regression',
        'num_leaves': 60,
        'subsample': 0.6143,
        'colsample_bytree': 0.6453,
        'min_split_gain': np.power(10, -2.5988),
        'reg_alpha': np.power(10, -2.2887),
        'reg_lambda': np.power(10, 1.7570),
        'min_child_weight': np.power(10, -0.1477),
        'verbose': -1,
        'seed': 3,
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'learning_rate': 0.05,
        'metric': 'rmse',
    }
    # Run KFold
    for trn_idx, val_idx in folds.split(data):
        # Train lightgbm
        clf = lgb.train(
            params=lgb_params,
            train_set=dtrain.subset(trn_idx),
            valid_sets=dtrain.subset(val_idx),
            num_boost_round=10000,
            early_stopping_rounds=100,
            verbose_eval=50
        )
        # Predict Out Of Fold and Test targets
        # Using lgb.train, predict will automatically select the best round for prediction
        oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
        sub_preds += clf.predict(test) / folds.n_splits
        # Display current fold score
        print(mean_squared_error(np.log1p(y['target'].iloc[val_idx]),
                                 oof_preds[val_idx]) ** .5)
    # Display Full OOF score (square root of a sum is not the sum of square roots)
    print('Full Out-Of-Fold score : %9.6f'
          % (mean_squared_error(np.log1p(y['target']), oof_preds) ** .5))

    return oof_preds, sub_preds


def main():
    # Get the data
    feats = ["select_features","statics"]
    
    data, test = get_data(feats)
    leak_test = get_leak_test()
    leak_indexes,_ = get_leak_indexes()
    _,non_ugly_indexes = get_leak_indexes()


    # Get target and ids
    y = data[['ID', 'target']].copy()
    del data['target'], data['ID']
    sub = test[['ID']].copy()
    del test['ID']

    # Free some memory
    gc.collect()

    # Predict test target
    oof_preds, sub_preds = fit_predict(data, y, test)

    # Store predictions
    #y['predictions'] = np.expm1(oof_preds)
    #y[['ID', 'target', 'predictions']].to_csv('reduced_set_oof.csv', index=False)

    #sub.loc[non_ugly_indexes,"target"] = leak_test.loc[non_ugly_indexes,"compiled_leak"]
    sub['target'] = np.expm1(sub_preds)
    
    name = "_".join(feats)
    sub[['ID', 'target']].to_csv('./output/{}_lgbm.csv'.format(name), index=False)


if __name__ == '__main__':
    main()