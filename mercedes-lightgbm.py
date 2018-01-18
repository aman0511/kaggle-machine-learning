import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD, PCA, FastICA
import lightgbm as lgbm
import matplotlib.pyplot as plt

train = pd.read_csv('mercedes_train.csv')
test = pd.read_csv('mercedes_test.csv')

extra = pd.read_csv('mercedes-extra.csv')
train_extra = extra.join(test, on='ID', how='inner', rsuffix='_bla')
train_extra.drop(['ID_bla'], axis=1, inplace=True)
train = pd.concat([train, train_extra])

y_train = train['y'].values
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
#df_all = pd.get_dummies(df_all, drop_first=True)

# Label-encoding object columns
for c in df_all.columns:
    if df_all[c].dtype == 'object':
        df_all[c] = LabelEncoder().fit_transform(df_all[c].values)


n_comp = 18

tsvd = TruncatedSVD(n_components=n_comp)
tsvd_results = tsvd.fit_transform(df_all)

pca = PCA(n_components=n_comp)
pca_results = pca.fit_transform(df_all)

ica = FastICA(n_components=n_comp)
ica_results = ica.fit_transform(df_all)

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
     df_all['pca_' + str(i)] = pca_results[:, i - 1]
     df_all['ica_' + str(i)] = ica_results[:, i - 1]
     df_all['tsvd_' + str(i)] = tsvd_results[:, i - 1]

train = df_all[:num_train]
test = df_all[num_train:]



# params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting': 'gbdt',
#     'learning_rate': 0.01 , #small learn rate, large number of iterations
#     'verbose': 0,
#     'num_leaves': 2 ** 5,
#     'bagging_fraction': 0.95,
#     'bagging_freq': 1,
#     'bagging_seed': RS,
#     'feature_fraction': 0.7,
#     'feature_fraction_seed': RS,
#     'max_bin': 100,
#     'max_depth': 7,
#     'num_rounds': ROUNDS,
# }


# Stolen from lightgbm starter 0.56+
##model = LGBMRegressor(boosting_type='gbdt', num_leaves=10, max_depth=4, learning_rate=0.005, n_estimators=675, max_bin=25, subsample_for_bin=50000, min_split_gain=0, min_child_weight=5, min_child_samples=10, subsample=0.995, subsample_freq=1, colsample_bytree=1, reg_alpha=0, reg_lambda=0, seed=0, nthread=-1, silent=True)


## Params for gbdt, no pca etc
# model = lgbm.LGBMRegressor(
#     objective='regression',
#     n_estimators=350,
#     learning_rate=0.015,
#     num_leaves=5,
#     min_data_in_leaf=40,
#     colsample_bytree=0.4,
#     min_gain_to_split=0.0004,
#     subsample=0.95,
#     subsample_freq=1
# )

## Params for dart + pca + ica + ...
model = lgbm.LGBMRegressor(
    boosting_type='dart',
    objective='regression',
    drop_rate=0.002,
    n_estimators=400,
    learning_rate=0.015,
    num_leaves=6,
    min_data_in_leaf=35,
    colsample_bytree=0.5,
    min_gain_to_split=0.0004,
    subsample=0.95,
    subsample_freq=1
)



# Xt, Xv, yt, yv = train_test_split(train, y_train, test_size=0.2)
# model.fit(Xt, yt, eval_set=[(Xv, yv), (Xt, yt)])
# lgbm.plot_metric(model)
# plt.show()
# exit()



model.fit(train, y_train)

lgbm.plot_importance(model)
plt.show()

df_sub = pd.DataFrame({'ID': id_test, 'y': model.predict(test)})
df_sub.to_csv('mercedes_submissions/lightgbm.csv', index=False)

exit()

# results = cross_val_score(model, train, y_train, scoring='r2', cv=5)
# print("R2 score: %.4f (+- %.4f)" % (results.mean(), results.std()))

## R2 score: 0.5616 (+- 0.0465) // Plain basic, get_dummies
## R2 score: 0.5640 (+- 0.0491) // Plain basic, labelencoder
## R2 score: 0.5649 (+- 0.0437) // With ica(8), pca(8), tsvd(8)
## R2 score: 0.5640 (+- 0.0446) // With ica(12), pca(12), tsvd(12)
## R2 score: 0.5641 (+- 0.0441) // With pca(24)


cv_pred = cross_val_predict(model, train, y_train, cv=5)
print("R2 score: %.4f" % r2_score(y_train, cv_pred))

## R2 score: 0.5533 // GBRT
## R2 score: 0.5526 // GBRT + ica(9), pca(9), tsvd(9)
## R2 score: 0.5536 // DART
## R2 score: 0.5532 // DART + ica(9), pca(9), tsvd(9)
## R2 score: 0.5532 // DART + ica(12), pca(12), tsvd(12)
## R2 score: 0.5541 // DART + ica(18), pca(18), tsvd(18)

exit()

##
## https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-tuning.md
##

gsc = GridSearchCV(
    estimator=model,
    param_grid={
        #'n_estimators': range(250,551,50),
        #'learning_rate': np.linspace(0.001, 0.05, 10),

        #'drop_rate': np.linspace(0.001, 0.004, 25)

        #'min_gain_to_split': np.linspace(0.0001, 0.002, 20),

        #'reg_alpha': np.linspace(0.1, 0.9, 9),
        #'reg_lambda': np.linspace(0.1, 0.9, 9),

        #'colsample_bytree': np.linspace(0.3, 0.9, 15),

        #'subsample': np.linspace(0.5, 1.0, 10),

        #'num_leaves': range(3,15),
        #'min_data_in_leaf': range(25,51,5),

        #'max_bin': range(100, 250, 25),
    },
    scoring='r2',
    cv=5
)

grid_result = gsc.fit(train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, train_mean, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))