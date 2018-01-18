import pandas as pd
import haversine
import numpy as np
from sklearn.preprocessing import RobustScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, InputLayer, GaussianNoise
import matplotlib.pyplot as plt


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
train = RobustScaler().fit_transform(train)

def rmse(actual, predicted):
    return K.sqrt(K.mean(K.square(actual - predicted)))

def build_model_fn(neurons=12, noise=0.25):
    model = Sequential()
    model.add(InputLayer(input_shape=(train.shape[1],)))
    model.add(GaussianNoise(noise))
    model.add(Dense(neurons, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=[rmse])
    return model

epochs = 15
plot_start = 0

model = build_model_fn()
history = model.fit(train,
                    train_y,
                    epochs=epochs,
                    verbose=2,
                    validation_split=.15)

dataz = pd.DataFrame({ 'rmse': history.history['rmse'][plot_start:],
                       'val_rmse': history.history['val_rmse'][plot_start:],
                       'epochs': range(plot_start+1,epochs+1) })
dataz.plot(x='epochs')
plt.show()
