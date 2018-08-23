# Santander_Value_Prediction (https://www.kaggle.com/c/santander-value-prediction-challenge)
====
My submission to the Kaggle competition "Santander_Value_Prediction", ranked 149th over 4551 teams(TOP4%).

I referred this article(https://amalog.hateblo.jp/entry/kaggle-feature-management) @SakuEji (his twitter)



# Code Description
## base.py
Basic functions and classes are included (get_input(),Feature etc) 

## features.py
Create feature and save it in feather format　
in commandline↓
`python features.py -f`

next feature description
### timespan
Divide columns directly connected to leak to the first half 10, 20, 30 etc. to make statistics


### statics
Statics of all columns

### subsets_statics
Statics of subsets,
I made subsets by leak columns set(not directly connected leak)



## lgbm.py 
Only learning

## get_leak.py
allmost ↓
https://www.kaggle.com/nulldata/jiazhen-to-armamut-via-gurchetan1000-0-56

## ipynb　etc
EDA,but I have not recorded much



# Good_Kernel

https://www.kaggle.com/sggpls/pipeline-kernel-xgb-fe-lb1-39
https://www.kaggle.com/alexpengxiao/preprocessing-model-averaging-by-xgb-lgb-1-39

