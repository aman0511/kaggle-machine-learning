import pandas, haversine
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBRegressor
from scipy.stats.distributions import uniform, randint
import numpy as np

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def preprocess(data):
    data['distance'] = data.apply(lambda r: haversine.haversine((r['pickup_latitude'], r['pickup_longitude']), (r['dropoff_latitude'], r['dropoff_longitude'])), axis=1)
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

train = pandas.read_csv('kaggle-nyc-train.csv', parse_dates=['pickup_datetime'], nrows=50000)
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

gsc = RandomizedSearchCV(
    estimator=XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=9, subsample=0.95),
    param_distributions={
        #'n_estimators': range(250, 1251, 250),
        #'learning_rate': uniform(0.05, 0.05),
        #'max_depth': randint(2, 12),
        'subsample': uniform(0.9, 0.1)
    },
    scoring=rmse,
    cv=5,
    verbose=2,
    n_iter=10
)

grid_result = gsc.fit(train, train_y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))