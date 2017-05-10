import numpy as np
from scipy.stats import skew
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler


#df_train = pd.read_csv("sberbank_train.csv", parse_dates=['timestamp'])
df_train = pd.read_csv("train_without_noise.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("sberbank_test.csv", parse_dates=['timestamp'])
#df_macro = pd.read_csv("sberbank_macro.csv", parse_dates=['timestamp'])
df_predict = pd.read_csv("sberbank-weekly-mean-price.csv", parse_dates=['timestamp'])

df_all = pd.concat([df_train, df_test])
df_all.drop(['id', 'price_doc'], axis=1, inplace=True)
#df_all = df_all.join(df_predict.set_index('timestamp'), on='timestamp', rsuffix='_mean')

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

xgb_models = []
for i in range(1,20):
    m = xgb.train(xgb_params, dtrain, num_boost_round=490)
    xgb_models.append(m)

#linear_model = ElasticNet(alpha=0.08)
#linear_model.fit(x_train, y_train)

# results = cross_val_score(linear_model, x_train.values, y_train, cv=10)
# print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
# exit()
# # Results: 0.2746 (0.0681) MSE

svr_model = SVR(kernel='rbf', C=1.5, epsilon=0.075)
svr_model.fit(x_train, y_train)

#svr_model2 = SVR(kernel='rbf', C=1.4, epsilon=0.07)
#svr_model2.fit(x_train, y_train)

#svr_model3 = SVR(kernel='rbf', C=1.6, epsilon=0.08)
#svr_model3.fit(x_train, y_train)

#y_test = xgb_model01.predict(dtest)

#y_test = np.expm1((linear_model.predict(x_test) + svr_model.predict(x_test)) / 2.0)
#y_test = ...

#y_test = np.expm1(linear_model.predict(x_test))

y_test = (np.expm1(svr_model.predict(x_test)) +
          #np.expm1(svr_model2.predict(x_test)) +
          #np.expm1(svr_model3.predict(x_test)) +
          xgb_model01.predict(dtest) +
          xgb_model02.predict(dtest) +
          xgb_model03.predict(dtest) +
          xgb_model04.predict(dtest) +
          xgb_model05.predict(dtest)) / 6.0


df_sub = pd.DataFrame({'id': df_test['id'], 'price_doc': y_test})
df_sub.to_csv('sberbank-submission-ensemble-svr-xgboosts.csv', index=False)


#
# https://www.kaggle.com/bguberfain/naive-xgb-lb-0-317
#


## https://www.kaggle.com/matthewa313/sberbankdatafix
## https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/32717