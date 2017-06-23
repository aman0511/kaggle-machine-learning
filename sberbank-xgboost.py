import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import random


#df_train = pd.read_csv("sberbank_train.csv", parse_dates=['timestamp'])
df_train = pd.read_csv("train_without_noise.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("sberbank_test.csv", parse_dates=['timestamp'])

# Drop crazy data points
df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)
df_train.drop(df_train[df_train["full_sq"] > 5000].index, inplace=True)

# ## SUBSAMPLING
# trainsub = df_train[df_train.timestamp < '2015-01-01']
# trainsub = trainsub[trainsub.product_type=="Investment"]
#
# ind_1m = trainsub[trainsub.price_doc <= 1000000].index
# ind_2m = trainsub[trainsub.price_doc == 2000000].index
# ind_3m = trainsub[trainsub.price_doc == 3000000].index
#
# train_index = set(df_train.index.copy())
#
# for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
#     ind_set = set(ind)
#     ind_set_cut = ind.difference(set(ind[::gap]))
#
#     train_index = train_index.difference(ind_set_cut)
#
# df_train = df_train.loc[train_index]
# ## SUBSAMPLING


y_train = df_train['price_doc'].values # * 0.95 + 10.05
id_test = df_test['id']

df_train.drop(['id'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
df_all = pd.concat([df_train, df_test])

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
# df_all['month_year'] = df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100
# df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# # House characteristics
# df_all['floor_from_top'] = df_all['max_floor'] - df_all['floor']
# df_all['rel_floor'] = df_all['floor'] / df_all['max_floor']
# df_all['avg_room_size'] = (df_all['life_sq'] - df_all['kitch_sq']) / df_all['num_room']
# df_all['prop_living'] = df_all['life_sq'] / df_all['full_sq']
# df_all['prop_kitchen'] = df_all['kitch_sq'] / df_all['full_sq']
# df_all['extra_area'] = df_all['full_sq'] - df_all['life_sq']
# df_all['age_at_sale'] = df_all['build_year'] - df_all.timestamp.dt.year
#
# df_all['ratio_preschool'] = df_all['children_preschool'] / df_all['preschool_quota']
# df_all['ratio_school'] = df_all['children_school'] / df_all['school_quota']

## Appartment building sales per month feature
# building_year_month = df_all['sub_area'] +\
#                       df_all['metro_km_avto'].astype(str) +\
#                       (df_all.timestamp.dt.month + \
#                        df_all.timestamp.dt.year * 100).astype(str)
# building_year_month_cnt_map = building_year_month.value_counts().to_dict()
# df_all['building_year_month_cnt'] = building_year_month.map(building_year_month_cnt_map)

## Apparetement building
# df_all['building'] = df_all['sub_area'] + df_all['metro_km_avto'].astype(str)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'price_doc'], axis=1, inplace=True)

df_all = pd.get_dummies(df_all)

x_train = df_all[:num_train]
x_test = df_all[num_train:]

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#                    verbose_eval=50, show_stdv=False)
# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# plt.show()
# print len(cv_output)
# exit()

# WITH mean price_doc
# [350]	train-rmse:1.9964e+06	test-rmse:2635520.0
# [400]	train-rmse:1.95806e+06	test-rmse:2631070.0
# [450]	train-rmse:1.92176e+06	test-rmse:2624400.0

# WITHOUT mean price_doc
# [500]	train-rmse:1.92084e+06	test-rmse:2.68534e+06
# [550]	train-rmse:1.88665e+06	test-rmse:2.68348e+06
# [600]	train-rmse:1.8565e+06	test-rmse:2.68095e+06

# WITH good/bad DOW + SUBSAMPLING
# [300]	train-rmse:1.64628e+06	test-rmse:2.38131e+06
# [350]	train-rmse:1.59425e+06	test-rmse:2.37554e+06
# [400]	train-rmse:1.54686e+06	test-rmse:2.37265e+06

# WITH house characteristics
# [450]	train-rmse:1.91022e+06	test-rmse:2.65082e+06
# [500]	train-rmse:1.87175e+06	test-rmse:2.64799e+06
# [550]	train-rmse:1.83872e+06	test-rmse:2.64573e+06

# WITH Also school characteristics
# [450]	train-rmse:1.91611e+06	test-rmse:2.65626e+06
# [500]	train-rmse:1.8761e+06	test-rmse:2.65304e+06
# [550]	train-rmse:1.84235e+06	test-rmse:2.65045e+06

# WITH Subsampling..
# [300]	train-rmse:1.47307e+06	test-rmse:2.39191e+06
# [350]	train-rmse:1.41965e+06	test-rmse:2.38503e+06
# [400]	train-rmse:1.37066e+06	test-rmse:2.38409e+06

model = xgb.train(xgb_params, dtrain, num_boost_round=368)

y_pred = model.predict(dtest)

df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv('sberbank-submission-xgboost.csv', index=False)

# xgb.plot_importance(model, max_num_features=50)
# plt.show()

# From:
#
# https://www.kaggle.com/bguberfain/sberbank-russian-housing-market/naive-xgb-lb-0-317/run/1113603
# https://www.kaggle.com/reynaldo/naive-xgb
# https://www.kaggle.com/philippsp/a-collection-of-new-features
#
# https://www.kaggle.com/c/sberbank-russian-housing-market/discussion/33269
#
# https://www.kaggle.com/asindico/predicting-house-prices/
#

# https://www.kaggle.com/yesemsanthoshkumar/fillna

##
## TODO
##
## https://www.kaggle.com/jasonbenner/lets-try-xgb10/code
##

for i in range(1,8):
    xgb_params.update(colsample_bytree=0.7 + random.uniform(-0.15, 0.15),
                      subsample=0.7 + random.uniform(-0.15, 0.15),
                      max_depth=5 - random.randint(0,1))
    m = xgb.train(xgb_params, dtrain, num_boost_round=400 + random.randint(-35,25))

    pd.DataFrame({'id': id_test, 'price_doc': m.predict(dtest)}) \
        .to_csv('sberbank_submissions/xgb-%d.csv' % i, index=False)

    print 'Trained XGB model %d' % i

##
## https://www.kaggle.com/aharless/attempt-to-reproduce-my-best-lb-score
##