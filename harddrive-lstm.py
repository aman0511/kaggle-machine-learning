import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, InputLayer, LSTM, Dropout

hdd = pd.read_csv('harddrive.csv')

hdd = hdd[hdd['model'].str.startswith('ST', na=False)]
hdd.sort_values(['serial_number', 'date'], ascending=True, inplace=True)

X = hdd[['smart_197_raw', 'smart_198_raw']][:50000]
y = hdd['failure'][:50000]

one_drive = hdd[hdd['serial_number'] == 'S30114J3']
X_one = one_drive[['smart_197_raw', 'smart_198_raw']]
y_one = one_drive['failure']

y = np.column_stack((
    y,
    1 - y
))

X.fillna(value=0, inplace=True)
X_one.fillna(value=0, inplace=True)

print X.shape

scaler = RobustScaler()
scaler.fit(X)
X = scaler.transform(X)
X_one = scaler.transform(X_one)

look_back = 10

# Transform into X=(t-look_back .. t) and Y=(t+1)
trainX = np.array([X[i:(i + look_back)] for i in range(len(X) - look_back - 1)])
predictX = np.array([X_one[i:(i + look_back)] for i in range(len(X_one) - look_back - 1)])
trainY = np.array([y[i + look_back] for i in range(len(y) - look_back - 1)])

epochs = 20

num_features = X.shape[1]

# Reshape inputs
trainX = np.reshape(trainX, (trainX.shape[0], num_features, look_back))
predictX = np.reshape(predictX, (predictX.shape[0], num_features, look_back))

model = Sequential()
model.add(InputLayer(input_shape=(num_features, look_back)))
model.add(LSTM(4)) #, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(4)) #, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
#model.add(Dense(1))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(trainX, trainY, epochs=epochs, batch_size=1024, verbose=2, class_weight={0: 1, 1: 10000})
#history = model.fit(trainX, trainY, epochs=epochs, batch_size=1024, verbose=2)

predictions = model.predict(predictX)

df_predict = pd.DataFrame({
    'smart_197_raw': one_drive['smart_197_raw'][look_back+1:],
    'actual': y_one[look_back+1:],
    #'predictions': predictions.flatten(),
    'predictions0': predictions[:,0].flatten(),
    'predictions1': predictions[:,1].flatten()
})
df_predict.plot()
plt.show()
