import numpy as np
from scipy.stats import skew
import pandas as pd
import xgboost as xgb
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

#df_train = pd.read_csv("sberbank_train.csv", parse_dates=['timestamp'])
df_train = pd.read_csv("train_without_noise.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("sberbank_test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("sberbank_macro.csv", parse_dates=['timestamp'])
#df_predict = pd.read_csv("sberbank-weekly-mean-price.csv", parse_dates=['timestamp'])

df_macro = df_macro[['timestamp','cpi', 'ppi', 'eurrub', 'usdrub']]

df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)
df_train.drop(df_train[df_train["full_sq"] > 5000].index, inplace=True)

df_all = pd.concat([df_train, df_test])
df_all.drop(['id', 'price_doc'], axis=1, inplace=True)
#df_all = df_all.join(df_predict.set_index('timestamp'), on='timestamp', rsuffix='_mean')
df_all = df_all.join(df_macro.set_index('timestamp'), on='timestamp', rsuffix='_macro')

# numeric_features = df_all.select_dtypes(include=[np.number])
# # numeric_features.dtypes
#
# corr = numeric_features.corr()
# #print corr['price_doc'].sort_values(ascending=False)[:100]
# print corr['price_doc']['cpi']
# print corr['price_doc']['ppi']
# print corr['price_doc']['eurrub']
# print corr['price_doc']['usdrub']
# exit()

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek
#df_all['bad_month'] = df_all['month'].apply(lambda x: 'Bad' if x in (7,10,11) else 'Good')
#df_all['bad_dow'] = df_all['dow'].apply(lambda x: 'Bad' if x in (5,6) else 'Good')
#df_all.drop(['month', 'dow', 'timestamp'], axis=1, inplace=True)
df_all.drop(['timestamp'], axis=1, inplace=True)

#df_all['rel_floor'] = df_all['floor'].astype(float) / df_all['max_floor'].astype(float)
#df_all['rel_kitch_sq'] = df_all['kitch_sq'].astype(float) / df_all['full_sq'].astype(float)

numeric_feats = df_all.dtypes[df_all.dtypes != "object"].index

num_train = len(df_train)
df_all = pd.get_dummies(df_all, drop_first=True)

x_train_unnormalized = df_all[:num_train]
x_test_unnormalized = df_all[num_train:]

df_all = df_all.fillna(df_all.mean())
#df_all = df_all.astype('float64')

# Log transform skewed features
skewness = df_all[numeric_feats].apply(lambda x: skew(x.dropna()))
left_skewed_feats = skewness[skewness > 0.5].index
right_skewed_feats = skewness[skewness < -0.5].index
df_all[left_skewed_feats] = np.log1p(df_all[left_skewed_feats])
#all_data[right_skewed_feats] = np.exp(all_data[right_skewed_feats])

scaler = RobustScaler()
df_all[numeric_feats] = scaler.fit_transform(df_all[numeric_feats])

x_train = df_all[:num_train]
x_test = df_all[num_train:]

y_train = np.log1p(df_train['price_doc'].values)
y_train_unnormalized = df_train['price_doc'].values


###
##  Make predictions
###

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train_unnormalized, y_train_unnormalized)
dtest = xgb.DMatrix(x_test_unnormalized)

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#                    verbose_eval=50, show_stdv=False)
# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# plt.show()
# print(len(cv_output))
# exit()

# [0]	train-rmse:8.13525e+06	test-rmse:8.13897e+06
# [50]	train-rmse:2.51758e+06	test-rmse:2.91121e+06
# [100]	train-rmse:2.18369e+06	test-rmse:2.74267e+06
# [150]	train-rmse:2.05612e+06	test-rmse:2.70521e+06
# [200]	train-rmse:1.96156e+06	test-rmse:2.68558e+06
# [250]	train-rmse:1.88872e+06	test-rmse:2.67151e+06
# [300]	train-rmse:1.82273e+06	test-rmse:2.66572e+06
# [350]	train-rmse:1.76323e+06	test-rmse:2.65945e+06
# [400]	train-rmse:1.71181e+06	test-rmse:2.65599e+06
# [450]	train-rmse:1.66511e+06	test-rmse:2.65458e+06
# 480

# xgb_model01 = xgb.train(xgb_params, dtrain, num_boost_round=490)
#
# xgb_model02 = xgb.train(xgb_params, dtrain, num_boost_round=440)
#
# xgb_params.update(colsample_bytree=0.6)
# xgb_model03 = xgb.train(xgb_params, dtrain, num_boost_round=500)
#
# xgb_params.update(subsample=0.6)
# xgb_model04 = xgb.train(xgb_params, dtrain, num_boost_round=500)
#
# xgb_params.update(max_depth=4)
# xgb_model05 = xgb.train(xgb_params, dtrain, num_boost_round=500)


# knn = KNeighborsRegressor(n_neighbors=25, weights='distance')
# knn.fit(x_train, y_train)
# pd.DataFrame({'id': df_test['id'], 'price_doc': np.expm1(knn.predict(x_test))})\
#     .to_csv('sberbank_submissions/knn.csv', index=False)
# print 'Trained KNN model'


#rf = RandomForestRegressor(n_estimators=250, min_samples_leaf=25)
# rf.fit(x_train, y_train)
# pd.DataFrame({'id': df_test['id'], 'price_doc': np.expm1(rf.predict(x_test))})\
#     .to_csv('sberbank_submissions/rf.csv', index=False)
# print 'Trained RF model'


# et = ExtraTreesRegressor(n_estimators=75, min_samples_leaf=10)
# et.fit(x_train, y_train)
# pd.DataFrame({'id': df_test['id'], 'price_doc': np.expm1(et.predict(x_test))})\
#         .to_csv('sberbank_submissions/et.csv', index=False)
# print 'Trained ET model'


# gsc = GridSearchCV(
#     # estimator=RandomForestRegressor(n_estimators=100, min_samples_leaf=25),
#     # param_grid={
#     #     'n_estimators': range(150,251,50), # 75
#     #     #'min_samples_leaf': (10,25,50), # 10
#     # },
#     estimator=KNeighborsRegressor(n_neighbors=25, weights='distance'),
#     param_grid={
#         #'n_neighbors': range(5,26,5),
#         'weights': ('uniform', 'distance')
#     },
#     cv=3
# )
# grid_result = gsc.fit(x_train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, test_stdev, train_mean, train_stdev, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['std_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['std_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f (%f) // Test : %f (%f) with: %r" % (train_mean, train_stdev, test_mean, test_stdev, param))
#
# exit()


# results = cross_val_score(rf, x_train.values, y_train, cv=3)
# print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
# exit()


# ElasticNet(alpha=0.08)              : 0.2716 (0.0654) MSE
# SVR(kernel=rbf,C=1.5,epsilon=0.075) : 0.2814 (0.0994) MSE
# RandomForestRegressor(...)          : 0.3311 (0.0624) MSE
# RF                                    0.3486 (with macro)
# KNearestNeighbours(...)             : 0.0879 (0.0590) MSE


# for i in range(1,5):
#     xgb_params.update(colsample_bytree=0.7 + random.uniform(-0.15, 0.15),
#                       subsample=0.7 + random.uniform(-0.15, 0.15),
#                       max_depth=5 - random.randint(0,1))
#     m = xgb.train(xgb_params, dtrain, num_boost_round=490 + random.randint(-35,25))
#
#     pd.DataFrame({'id': df_test['id'], 'price_doc': m.predict(dtest)}) \
#         .to_csv('sberbank_submissions/xgb-macro-%d.csv' % i, index=False)
#
#     print 'Trained XGB model %d' % i

# linear_model = ElasticNet(alpha=0.08)
# linear_model.fit(x_train, y_train)
# pd.DataFrame({'id': df_test['id'], 'price_doc': np.expm1(linear_model.predict(x_test))}) \
#     .to_csv('sberbank_submissions/elasticnet.csv', index=False)
# print 'Trained ElasticNet() model!'
#
# svr_model = SVR(kernel='rbf', C=1.5, epsilon=0.075)
# svr_model.fit(x_train, y_train)
# pd.DataFrame({'id': df_test['id'], 'price_doc': np.expm1(svr_model.predict(x_test))}) \
#     .to_csv('sberbank_submissions/svr-1.csv', index=False)
# print 'Trained SVM 1!'
#
# svr_model2 = SVR(kernel='rbf', C=1.4, epsilon=0.07)
# svr_model2.fit(x_train, y_train)
# pd.DataFrame({'id': df_test['id'], 'price_doc': np.expm1(svr_model2.predict(x_test))}) \
#     .to_csv('sberbank_submissions/svr-2.csv', index=False)
# print 'Trained SVM 2!'


#svr_model3 = SVR(kernel='rbf', C=1.6, epsilon=0.08)
#svr_model3.fit(x_train, y_train)

#y_test = xgb_model01.predict(dtest)

#y_test = np.expm1((linear_model.predict(x_test) + svr_model.predict(x_test)) / 2.0)
#y_test = ...

#y_test = np.expm1(linear_model.predict(x_test))

# y_test = (np.expm1(svr_model.predict(x_test)) +
#           #np.expm1(svr_model2.predict(x_test)) +
#           #np.expm1(svr_model3.predict(x_test)) +
#           xgb_model01.predict(dtest) +
#           xgb_model02.predict(dtest) +
#           xgb_model03.predict(dtest) +
#           xgb_model04.predict(dtest) +
#           xgb_model05.predict(dtest)) / 6.0

# y_test = np.zeros(len(x_test))

# y_test = np.expm1((svr_model.predict(x_test) +
#                    svr_model2.predict(x_test) +
#                    linear_model.predict(x_test)) / 3.0) * 2.0
#
# for m in xgb_models:
#     y_test = y_test + m.predict(dtest)
#
# y_test = y_test / float(len(xgb_models) + 2.0)
#
# df_sub = pd.DataFrame({'id': df_test['id'], 'price_doc': y_test})
# df_sub.to_csv('sberbank-submission-ensemble-svr-xgboosts.csv', index=False)


#
# https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317
#


## https://www.kaggle.com/matthewa313/sberbankdatafix
## https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32717


## https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.regressor/#stackingregressor

## https://www.kaggle.com/remap1/exploring-the-volatility-of-the-economy