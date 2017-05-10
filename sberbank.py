import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor


df_train = pd.read_csv("sberbank_train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("sberbank_test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("sberbank_macro.csv", parse_dates=['timestamp'])

# id_test = df_test['id']
#
# y_train = df_train["price_doc"]
# x_train = df_train.drop(["id", "timestamp", "price_doc"], axis=1)
# x_test = df_test.drop(["id", "timestamp"], axis=1)

#can't merge train with test because the kernel run for very long time
#
# for c in x_train.columns:
#     if x_train[c].dtype == 'object':
#         lbl = LabelEncoder()
#         lbl.fit(list(x_train[c].values))
#         x_train[c] = lbl.transform(list(x_train[c].values))
#         #x_train.drop(c,axis=1,inplace=True)
#
# for c in x_test.columns:
#     if x_test[c].dtype == 'object':
#         lbl = LabelEncoder()
#         lbl.fit(list(x_test[c].values))
#         x_test[c] = lbl.transform(list(x_test[c].values))
#         #x_test.drop(c,axis=1,inplace=True)

#log transform the price:
#df_train["price_doc"] = np.log1p(df_train["price_doc"])

#df_train['price_doc'].hist(bins=50)
#plt.show()

# Drop crazy data points
df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)
df_train.drop(df_train[df_train["full_sq"] > 5000].index, inplace=True)
importance = sorted(importance.items(), key=operator.itemgetter(1))
y_train = df_train['price_doc'].values
id_test = df_test['id']

#df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_train.drop(['id'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# Build df_all = (df_train+df_test).join(df_macro)
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
#df_all = df_all.join(df_macro.set_index('timestamp'), on='timestamp', rsuffix='_macro')

# Add month-year
#month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
#month_year_cnt_map = month_year.value_counts().to_dict()
#df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
#week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
#week_year_cnt_map = week_year.value_counts().to_dict()
#df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month_year, month and day-of-week
# df_all['week_year'] = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
# df_all['month_year'] = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
# df_all['month'] = df_all.timestamp.dt.month
#df_all['dow'] = df_all.timestamp.dt.dayofweek

#df_all['bad_month'] = df_all['month'].apply(lambda x: x in (7,10,11))
#df_all['bad_dow'] = df_all['dow'].apply(lambda x: x in (5,6))

#df_all.drop(['month', 'dow'], axis=1, inplace=True)
#df_all.drop(['dow'], axis=1, inplace=True)


# print df_all.groupby(['week_year'])['price_doc'].mean()
# print df_all.groupby(['month_year'])['price_doc'].mean()
# print df_all.groupby(['month'])['price_doc'].mean()
# print df_all.groupby(['bad_month'])['price_doc'].mean()
# print df_all.groupby(['dow'])['price_doc'].mean()
# print df_all.groupby(['bad_dow'])['price_doc'].mean()

# df_all.groupby(['timestamp'])['price_doc'].mean().plot()
# plt.show()
# exit()

#mean_price = df_all.groupby(['timestamp'])['price_doc'].mean()
#df_all = df_all.join(mean_price, on='timestamp', rsuffix='_mean')
# print df_all[['timestamp', 'price_doc', 'price_doc_mean']].head(n=50)
# exit()

# Other feature engineering
#df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
#df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

# Remove timestamp column (may overfit the model in train)
#df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)
df_all.drop(['timestamp'], axis=1, inplace=True)
df_all.drop(['price_doc'], axis=1, inplace=True)

#numeric_feats = df_all.dtypes[df_all.dtypes != "object"].index

df_all = pd.get_dummies(df_all)
#df_all = df_all.fillna(df_all.mean())
#df_all = df_all.astype('float64')

#scaler = RobustScaler()
#df_all[numeric_feats] = scaler.fit_transform(df_all[numeric_feats])

x_train = df_all[:num_train]
x_test = df_all[num_train:]


# gsc = GridSearchCV(
#     estimator=xgb.XGBRegressor(n_estimators=325, max_depth=3, subsample=0.7, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=0.8),
#     param_grid={
#         'n_estimators': range(275,350,25),
#         #'learning_rate': (0.02, 0.04, 0.06, 0.08, 0.10),
#         #'subsample': (0.6, 0.7, 0.8),
#         #'colsample_bytree': (0.4, 0.5, 0.6),
#         #'max_depth': range(3,5),
#         #'reg_alpha': (0.0, 0.1, 0.2, 0.3),
#         #'reg_lambda': (0.8, 0.9, 1.0),
#         #'gamma': (0, 0.005, 0.01),
#     },
#     cv=5
# )
# grid_result = gsc.fit(X_train, y_train)
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


xgb_params = {
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#                    verbose_eval=50, show_stdv=False)
# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# plt.show()
#
# exit()

# WITH mean price_doc
# [350]	train-rmse:1.9964e+06	test-rmse:2635520.0
# [400]	train-rmse:1.95806e+06	test-rmse:2631070.0
# [450]	train-rmse:1.92176e+06	test-rmse:2624400.0

# WITHOUT mean price_doc
# [500]	train-rmse:1.92084e+06	test-rmse:2.68534e+06
# [550]	train-rmse:1.88665e+06	test-rmse:2.68348e+06
# [600]	train-rmse:1.8565e+06	test-rmse:2.68095e+06

# WITH good/bad DOW
# [350]	train-rmse:2.03963e+06	test-rmse:2.67462e+06
# [400]	train-rmse:1.99466e+06	test-rmse:2.66868e+06
# [450]	train-rmse:1.95785e+06	test-rmse:2.6652e+06

model = xgb.train(xgb_params, dtrain, num_boost_round=800)

print model.feature_importances_

#fig, ax = plt.subplots(1, 1, figsize=(8, 13))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
#plt.show()

y_pred = model.predict(dtest)

#df_sub = pd.DataFrame({'id': id_test, 'price_doc': np.expm1(y_pred)})
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv('sberbank-submission.csv', index=False)

#
# From:
#
# https://www.kaggle.com/bguberfain/sberbank-russian-housing-market/naive-xgb-lb-0-317/run/1113603
# https://www.kaggle.com/reynaldo/naive-xgb
#