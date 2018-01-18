from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import make_scorer
from scipy.stats.distributions import uniform, randint
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import pandas as pd

df_partial = pd.read_csv('porto-train.csv', nrows=50000)
df_train = pd.read_csv('porto-train.csv', na_values=-1)
df_test = pd.read_csv('porto-test.csv', na_values=-1)
df_sample = pd.read_csv('porto-sample_submission.csv')

col_to_drop = df_train.columns[df_train.columns.str.startswith('ps_calc_')]
df_partial = df_partial.drop(col_to_drop, axis=1)
df_train = df_train.drop(col_to_drop, axis=1)
df_test = df_test.drop(col_to_drop, axis=1)

X_partial = df_partial.drop(['id', 'target'], axis=1)
Y_partial = df_partial['target']

X = df_train.drop(['id', 'target'], axis=1)
Y = df_train['target']

##
## https://www.kaggle.com/rshally/porto-xgb-lgb-kfold-lb-0-282
##
## https://www.kaggle.com/hireme/start-with-lightgbm-0-27
##

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', -1.0 * gini(y, pred) / gini(y, y)

def gini_lgb(truth, predictions):
    score = gini(truth, predictions) / gini(truth, truth)
    return 'gini', score, True

def gini_sklearn(truth, predictions):
    return -1.0 * gini(truth, predictions) / gini(truth, truth)

model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9
)

Xtrain, Xeval, Ytrain, Yeval = train_test_split(X, Y, test_size=0.2)

model.fit(
    Xtrain, Ytrain,
    eval_set=[(Xtrain, Ytrain), (Xeval, Yeval)],
    eval_metric=gini_xgb,
    early_stopping_rounds=100,
    verbose=50
)



# df_sample['target'] = model.predict(
#     xgb.DMatrix(df_test.drop('id', axis=1)),
#     ntree_limit=model.best_ntree_limit+25
# )
# df_sample.to_csv('porto-submission-xgb.csv', index=False, float_format='%.5f')

exit()

gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)

rsc = RandomizedSearchCV(
     estimator=model,
     param_distributions={
         #'n_estimators': randint(25, 250),
         #'subsample': uniform(0.5, 0.5),
         #'subsample_freq': randint(2, 25),
         #'colsample_bytree': uniform(0.5, 0.5),
         #'learning_rate': uniform(0.0, 0.1),
         #'min_child_samples': randint(5, 500),
         'num_leaves': randint(5, 200),
     },
     scoring=gini_scorer,
     cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2),
     verbose=2,
     n_iter=5
)

grid_result = rsc.fit(X_partial, Y_partial)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, train_mean, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))
