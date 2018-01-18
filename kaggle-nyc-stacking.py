import pandas, haversine
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from pysmac.optimize import fmin
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
import numpy as np
from catboost import CatBoostRegressor
from mlxtend.regressor import StackingCVRegressor

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def manhattan_distance(lat1, lon1, lat2, lon2):
    return np.absolute(lon1 - lon2) + np.absolute(lat1 - lat2)

def preprocess(data):
    data['distance'] = data.apply(lambda r: haversine.haversine((r['pickup_latitude'], r['pickup_longitude']), (r['dropoff_latitude'], r['dropoff_longitude'])), axis=1)

    data['manhattan_distance'] = manhattan_distance(
        data['pickup_latitude'].values, data['pickup_longitude'].values,
        data['dropoff_latitude'].values, data['dropoff_longitude'].values)

    data['center_latitude'] = data.apply(lambda r: (r['pickup_latitude'] + r['dropoff_latitude']) / 2.0, axis=1)
    data['center_longitude'] = data.apply(lambda r: (r['pickup_longitude'] + r['dropoff_longitude']) / 2.0, axis=1)

    data['direction'] = bearing_array(
        data['pickup_latitude'].values, data['pickup_longitude'].values,
        data['dropoff_latitude'].values, data['dropoff_longitude'].values)

    data['month'] = data.pickup_datetime.dt.month
    data['day'] = data.pickup_datetime.dt.day
    data['dw'] = data.pickup_datetime.dt.dayofweek
    data['h'] = data.pickup_datetime.dt.hour

    data['store_and_fwd_flag'] = data['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)

    return data.drop(['id', 'pickup_datetime'], axis=1)

#train = pandas.read_csv('kaggle-nyc-train.csv', parse_dates=['pickup_datetime'], nrows=200000)
train = pandas.read_csv('kaggle-nyc-train.csv', parse_dates=['pickup_datetime'])
test = pandas.read_csv('kaggle-nyc-test.csv', parse_dates=['pickup_datetime'])

train_y = np.log1p(train['trip_duration'].values)
test_ids = test['id']

train = train.drop(['dropoff_datetime', 'trip_duration'], axis=1)

train = preprocess(train).values
test = preprocess(test).values

# xgb_params = {
#     #'eta': 0.05,
#     'subsample': 0.75,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     'silent': 1
# }
#
# cv_output = xgb.cv(xgb_params, xgb.DMatrix(train, train_y), num_boost_round=500, #early_stopping_rounds=50,
#                    verbose_eval=50, show_stdv=False)
# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# plt.show()
# print len(cv_output)
# exit()

def rmsle_fun(predicted, actual):
    #return -1.0 * np.sqrt(np.mean(np.square(np.log1p(predicted) - np.log1p(actual))))
    return np.sqrt(np.mean(np.square(np.log1p(predicted) - np.log1p(actual))))

def rmse_fun(predicted, actual):
    #return -1.0 * np.sqrt(np.mean(np.square(predicted - actual)))
    return np.sqrt(np.mean(np.square(predicted - actual)))

rmlse = make_scorer(rmsle_fun, greater_is_better=False)
rmse = make_scorer(rmse_fun, greater_is_better=False)

# def objective(x):
#     #return -1.0 * cross_val_score(Ridge(alpha=x[0]), train, train_y, scoring=rmse).mean()
#     return -1.0 * cross_val_score(Lasso(alpha=x[0]), train, train_y, scoring=rmse).mean()
#     #return -1.0 * cross_val_score(SVR(C=x[0]), train, train_y, scoring=rmse).mean()
#
# parameters, score = fmin(objective=objective,
#                          x0=[0.01],
#                          xmin=[0.0001],
#                          xmax=[100.0],
#                          max_evaluations=25)
#
# print('Lowest function value found: %f' % score)
# print('Parameter setting %s' % parameters)
#
# exit()


class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors
        self.predictions = None

    def fit(self, X, y):
        for regr in self.regressors:
            regr.fit(X, y)
        return self

    def predict(self, X):
        self.predictions = np.column_stack([regr.predict(X) for regr in self.regressors])
        return np.mean(self.predictions, axis=1)

lgbm_model = LGBMRegressor(
    n_estimators=1200,
    subsample=0.95,
    subsample_freq=20,
    learning_rate=0.05,
    num_leaves=100,
    min_data_in_leaf=50
)

catboost_model = CatBoostRegressor(iterations=1000)
xgb_model = XGBRegressor(n_estimators=750, max_depth=9, learning_rate=0.05, subsample=0.9)
rf_model = RandomForestRegressor(n_estimators=75, min_samples_leaf=15, min_samples_split=25)

averaged = AveragingRegressor([catboost_model, xgb_model, rf_model, lgbm_model])

stacked = StackingCVRegressor(
    regressors=[xgb_model, rf_model, lgbm_model],
    meta_regressor=Ridge()
)

# models = [
#     ('DecisionTree', DecisionTreeRegressor(min_samples_leaf=20, min_samples_split=30)),
#     ('RandomForest', rf_model),
#     ('CatBoost    ', catboost_model),
#     ('LightGBM    ', lgbm_model),
#     ('XGBoost     ', xgb_model),
#     #('Ridge       ', Ridge(alpha=75.0)),
#     #('Lasso       ', Lasso(alpha=0.75)),
#     #('SVR         ', SVR(C=75.0)),
#     #('KNN         ', KNeighborsRegressor(n_neighbors=25, weights='distance')),
#     ('Stack       ', stacked),
#     ('Averaged    ', averaged)
# ]
#
# for name, model in models:
#     score = cross_val_score(model, train, train_y, scoring=rmse)
#     print("Model %s Score: %.4f (+- %.4f)" % (name, score.mean(), score.std()))
#
# exit()

xgb_model.fit(train, train_y)

result = pandas.DataFrame({'id': test_ids, 'trip_duration': np.expm1(xgb_model.predict(test))})
result['trip_duration'] = result['trip_duration'].clip(0.,9999999999.)
result.to_csv('kaggle-nyc-submission.csv', index=False)


## Take more features and ideas from ...
## https://www.kaggle.com/misfyre/stacking-model-378-lb-375-cv

# Model DecisionTree Score: -0.4319 (+- 0.0003)
# Model RandomForest Score: -0.3965 (+- 0.0003)
# Model CatBoost     Score: -0.4046 (+- 0.0004)
# Model LightGBM     Score: -0.4014 (+- 0.0002)
# Model XGBoost      Score: -0.3823 (+- 0.0003)
# Model Stack        Score: -0.3820 (+- 0.0003)
# Model Averaged     Score: -0.3884 (+- 0.0002)