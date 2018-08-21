# コードの説明
## base.py
get_input()などの基本的な関数,Featureなどのクラスが入ってる

## features.py
特徴量を記述,コマンドラインで実行するとfeather形式で保存
以下特徴量の説明

### timespan
leakに直結する40columnsを機関で分けて統計量に変換する

### statics
全columnsの統計量

### subsets_statics
グループ別での統計量、leakのグループをそのまま使った



## lgbm.py 
学習を回す

## get_leak.py
leakを見つけるプログラム(kernel丸パクリ）

## そのほかipynb
EDAなど（適当に管理しすぎてあまり残ってない）
後で消すかも





# Santander

https://www.kaggle.com/c/santander-value-prediction-challenge




# 良かったKernel

https://www.kaggle.com/sggpls/pipeline-kernel-xgb-fe-lb1-39

https://www.kaggle.com/nulldata/jiazhen-to-armamut-via-gurchetan1000-0-56

https://www.kaggle.com/alexpengxiao/preprocessing-model-averaging-by-xgb-lgb-1-39


# 良かった文献


https://amalog.hateblo.jp/entry/kaggle-feature-management

# とりあえず自分でやったこと

PCA,tSVD,RandomProjecionで4000以上の特徴量をまとめた
全特徴の平均・合計などの統計量を追加した、
subsets出の特徴量を追加した。
学習を回して寄与度が少ないものを削除して再学習したものを提出

stackingはあまりやる気がしなかったのでsinglemodelで提出した。やっておけばよかったと後悔。

# 順位
149/4555th(top4%)
