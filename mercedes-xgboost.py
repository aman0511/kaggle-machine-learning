import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA, FastICA
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

train = pd.read_csv('mercedes_train.csv')
test = pd.read_csv('mercedes_test.csv')

y_train = train['y'].values
y_mean = np.mean(y_train)
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

n_comp = 12

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results = tsvd.fit_transform(df_all)

pca = PCA(n_components=n_comp, random_state=420)
pca_results = pca.fit_transform(df_all)

ica = FastICA(n_components=n_comp, random_state=420)
ica_results = ica.fit_transform(df_all)

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results = grp.fit_transform(df_all)

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results = srp.fit_transform(df_all)

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    df_all['pca_' + str(i)] = pca_results[:, i - 1]
    df_all['ica_' + str(i)] = ica_results[:, i - 1]
    df_all['tsvd_' + str(i)] = tsvd_results[:, i - 1]
    df_all['grp_' + str(i)] = grp_results[:, i - 1]
    df_all['srp_' + str(i)] = srp_results[:, i - 1]

train = df_all[:num_train]
test = df_all[num_train:]

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

# model = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.0045, subsample=0.93,
#                                  objective='reg:linear', n_estimators=1300)
#
# model.fit(train, y_train)
#
# y_test = model.predict(test)


xgb_params = {
    #'n_trees': 520,
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.93,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1500, early_stopping_rounds=50,
#                    verbose_eval=50, show_stdv=False)
# # cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# # plt.show()
# print len(cv_output)
# exit()

model = xgb.train(xgb_params, dtrain, num_boost_round=650)
y_test = model.predict(dtest)



df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('mercedes-submission.csv', index=False)

# Xt, Xv, Yt, Yv = train_test_split(train, y_train, test_size=0.25)
# model.fit(Xt, Yt, [(Xv,Yv)], eval_metric='rmse', early_stopping_rounds=50)

# gsc = GridSearchCV(
#     estimator=model,
#     param_grid={
#         'max_depth': range(3,5),
#         #'subsample': (0.8, 0.9, 1.0),
#         #'colsample_bytree': (0.3, 0.4, 0.5),
#     },
#     scoring='r2',
#     cv=5
# )
#
# grid_result = gsc.fit(train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, train_mean, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))

