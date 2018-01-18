import pandas as pd
import haversine
from sklearn.metrics import make_scorer
from sklearn.model_selection import validation_curve, learning_curve, RandomizedSearchCV
from scipy.stats.distributions import uniform, randint
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

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

train = pd.read_csv('kaggle-nyc-train.csv', parse_dates=['pickup_datetime'], nrows=250000)
#train = pd.read_csv('kaggle-nyc-train.csv', parse_dates=['pickup_datetime'])

train_y = np.log1p(train['trip_duration'])

train = train.drop(['dropoff_datetime', 'trip_duration'], axis=1)

train = preprocess(train)

def rmse_fun(predicted, actual):
    return np.sqrt(np.mean(np.square(predicted - actual)))

rmse = make_scorer(rmse_fun, greater_is_better=False)

model = RandomForestRegressor(
    n_estimators=75,
    min_samples_leaf=15,
    min_samples_split=25
)

# gsc = RandomizedSearchCV(
#     estimator=model,
#     param_distributions={
#         #'n_estimators': randint(50, 750),
#         'min_samples_leaf': randint(10, 30),
#         'min_samples_split': randint(20, 40),
#         #'max_depth': randint(3, 15),
#     },
#     scoring=rmse,
#     cv=3,
#     verbose=2,
#     n_jobs=4,
#     n_iter=10
# )
#
# grid_result = gsc.fit(train, train_y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, train_mean, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))
#
# exit()

# train_sizes, train_scores, valid_scores = learning_curve(model, train, train_y, cv=3, scoring=rmse,
#                                                          train_sizes=np.linspace(0.1, 1.0, 12))
#
# dataz = pd.DataFrame({ 'Training Sizes': train_sizes,
#                        'Training Scores': np.mean(train_scores, axis=1),
#                        'Validation Scores': np.mean(valid_scores, axis=1) })
# dataz.plot(x='Training Sizes')
# plt.show()

# param_range = range(25, 176, 50)
# train_scores, test_scores = validation_curve(model, train, train_y, cv=3, scoring=rmse,
#                                              param_name='n_estimators', param_range=param_range)
#
# dataz = pd.DataFrame({ 'Trees': param_range,
#                        'Training Scores': np.mean(train_scores, axis=1),
#                        'Validation Scores': np.mean(test_scores, axis=1) })
# dataz.plot(x='Trees')
# plt.show()
