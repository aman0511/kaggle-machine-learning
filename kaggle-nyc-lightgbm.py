import pandas, haversine
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV
from lightgbm.sklearn import LGBMRegressor
from lightgbm.plotting import plot_importance
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

train = pandas.read_csv('kaggle-nyc-train.csv', parse_dates=['pickup_datetime'], nrows=200000)
#train = pandas.read_csv('kaggle-nyc-train.csv', parse_dates=['pickup_datetime'])
#test = pandas.read_csv('kaggle-nyc-test.csv', parse_dates=['pickup_datetime'])

train_y = np.log1p(train['trip_duration'])
#test_ids = test['id']

train = train.drop(['dropoff_datetime', 'trip_duration'], axis=1)

train = preprocess(train)
#test = preprocess(test)

def rmse_fun(predicted, actual):
    return np.sqrt(np.mean(np.square(predicted - actual)))

rmse = make_scorer(rmse_fun, greater_is_better=False)

model = LGBMRegressor(
    n_estimators=1200,
    subsample=0.95,
    subsample_freq=20,
    learning_rate=0.05,
    num_leaves=100,
    min_data_in_leaf=50
)

# gsc = GridSearchCV(
#     estimator=model,
#     param_grid={
#         #'n_estimators': range(650,1051,100),
#         #'num_leaves': range(20,91,10),
#         #'subsample': np.linspace(0.75, 0.96, 5),
#         #'subsample_freq': range(3,26,3),
#         #'learning_rate': np.linspace(0.01, 0.15, 7),
#         #'min_data_in_leaf': range(25,66,5),
#     },
#     scoring=rmse,
#     cv=5
# )
# grid_result = gsc.fit(train, train_y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, train_mean, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))
# exit()

# score = cross_val_score(model, train.values, train_y.values, scoring=rmse)
# print("Score: %.4f (+- %.4f)" % (score.mean(), score.std()))
# exit()

model.fit(train, train_y)

# result = pandas.DataFrame({'id': test_ids, 'trip_duration': np.expm1(model.predict(test))})
# result['trip_duration'] = result['trip_duration'].clip(0.,9999999999.)
# result.to_csv('kaggle-nyc-submission.csv', index=False)


plot_importance(model, max_num_features=200)
plt.show()
