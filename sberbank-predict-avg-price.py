from operator import itemgetter
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense, InputLayer, LSTM, Dropout

df_train = pd.read_csv("sberbank_train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("sberbank_test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("sberbank_macro.csv", parse_dates=['timestamp'])

df_train['week_year'] = (df_train.timestamp.dt.weekofyear + df_train.timestamp.dt.year * 100)
df_macro['week_year'] = (df_macro.timestamp.dt.weekofyear + df_macro.timestamp.dt.year * 100)

mean_price = df_train.groupby(['week_year'])['price_doc'].mean()
df_macro = df_macro.join(mean_price, on='week_year', how='outer', rsuffix='_mean')
# df_macro[['timestamp', 'price_doc']].plot()
# plt.show()

# numeric_feats = df_macro.dtypes[df_macro.dtypes == "float64"].index
# correlations = [(col, math.fabs(df_macro['price_doc'].corr(df_macro[col]))) for col in numeric_feats]
# correlations = sorted(correlations, key=itemgetter(1))
# for corr in correlations:
#     print corr
# exit()

#
# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# https://groups.google.com/forum/#!topic/keras-users/9GsDwkSdqBg
# http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
#

#
# Plan of attack :
# (X) Check average prices improve predictions (YES THEY DO)
# (X) Make macro dataframe with average prices (from training data)
# (X) Data prep / handle n/a's
# (X) Try to predict average prices from macro features
# (X) Merge predictions back to train/test data and see if that improves our score
#

datasetY = df_macro['price_doc'][596:2010]
datasetY = datasetY.fillna(datasetY.mean()).values

timestamps = df_macro['timestamp'].values

# Drop columns we don't want to learn from
df_macro.drop(['price_doc', 'timestamp', 'week_year'], axis=1, inplace=True)

df_macro = pd.get_dummies(df_macro)

#datasetX = df_macro['gdp_quart'][596:2010]
# datasetX = pd.DataFrame({'oil_urals': df_macro['oil_urals'][596:2010],
#                          'gdp_quart': df_macro['gdp_quart'][596:2010],
#                          'usdrub': df_macro['usdrub'][596:2010],
#                          'eurrub': df_macro['eurrub'][596:2010],
#                          'mortgage_value': df_macro['mortgage_value'][596:2010],
#                          'mortgage_growth': df_macro['mortgage_growth'][596:2010],
#                          'mortgage_rate': df_macro['mortgage_rate'][596:2010]
#                          })

# Keep highest correlated columns
df_macro = df_macro[['bandwidth_sports', 'gdp_annual_growth', 'load_of_teachers_school_per_teacher',
                     'cpi', 'fixed_basket', 'provision_nurse', 'salary', 'gdp_deflator', 'labor_force',
                     'turnover_catering_per_cap', 'gdp_annual', 'ppi', 'average_life_exp',
                     'employment', 'eurrub', 'usdrub']]

datasetX = df_macro[596:2010]
the_mean = datasetX.mean()
datasetX = datasetX.fillna(datasetX.mean()).values

# Intentionall filling N/A's with the values from the limited dataset
datasetX_predict = df_macro[596:].fillna(the_mean).values

scalerX = RobustScaler()
datasetX = scalerX.fit_transform(datasetX)
datasetX_predict = scalerX.transform(datasetX_predict)

scalerY = RobustScaler()
datasetY = scalerY.fit_transform(datasetY)
1
look_back = 90

# Transform into X=(t-look_back .. t) and Y=(t+1)
trainX = np.array([datasetX[i:(i + look_back)] for i in range(len(datasetX) - look_back - 1)])
predictX = np.array([datasetX_predict[i:(i + look_back)] for i in range(len(datasetX_predict) - look_back - 1)])
trainY = np.array([datasetY[i + look_back] for i in range(len(datasetY) - look_back - 1)])

epochs = 50

num_features = datasetX.shape[1]

# Reshape inputs
trainX = np.reshape(trainX, (trainX.shape[0], num_features, look_back))
predictX = np.reshape(predictX, (predictX.shape[0], num_features, look_back))

model = Sequential()
model.add(InputLayer(input_shape=(num_features, look_back)))
model.add(LSTM(12))
model.add(Dropout(0.3))
model.add(Dense(12))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# dataz = pd.DataFrame({ 'loss': history.history['loss'],
#                        'epochs': range(1,epochs+1) })
# dataz.plot(x='epochs')
# plt.show()
# exit()

predictions = model.predict(predictX)
#predictions = model.predict(trainX)
predictions = predictions.flatten()

trainY = scalerY.inverse_transform(trainY)
predictions = scalerY.inverse_transform(predictions)

#trainY = np.append(np.zeros(596), trainY)
trainY = np.append(trainY, np.zeros(474))

df_predict = pd.DataFrame({
    'actual': trainY,
    'predictions': predictions
})
df_predict.plot()
plt.show()

df_predict = pd.DataFrame({
    'weekly_mean_price': predictions,
    'timestamp': timestamps[596+look_back+1:]
})
df_predict.to_csv('sberbank-weekly-mean-price.csv', index=False)


