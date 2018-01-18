import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from multiprocessing import *

train = pd.read_csv('porto-train.csv')
test = pd.read_csv('porto-test.csv')
col = [c for c in train.columns if c not in ['id','target']]

d_median = train.median(axis=0)

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_cat_13xps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        df[c+str('_mean_range')] = (df[c].values > d_median[c]).astype(int)
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = sklearn.model_selection.train_test_split(train, train['target'], test_size=0.25, random_state=99)

#keep dist
x1 = multi_transform(x1)
y1 = x1['target']
x2 = multi_transform(x2)
y2 = x2['target']
test = multi_transform(test)

col = [c for c in x1.columns if c not in ['id','target']] #oopppss
x1 = x1[col]
x2 = x2[col]

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit+45)
test[['id','target']].to_csv('porto-submission.csv', index=False, float_format='%.5f')

#Extras
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (7.0, 12.0)
xgb.plot_importance(booster=model)
plt.show()
