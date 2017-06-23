import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, InputLayer, GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt

train = pd.read_csv('mercedes_train.csv')
test = pd.read_csv('mercedes_test.csv')

y_train = train['y'].values
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

scaler = RobustScaler()
df_all = scaler.fit_transform(df_all)

train = df_all[:num_train]
test = df_all[num_train:]

# Keep only the most contributing features
sfm = SelectFromModel(Lasso(alpha=0.03))
sfm.fit(train, y_train)
train = sfm.transform(train)
test = sfm.transform(test)

print 'Number of features : %d' % train.shape[1]

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_model_fn(neurons=24, noise=0.25):
    model = Sequential()
    model.add(InputLayer(input_shape=(train.shape[1],)))
    model.add(GaussianNoise(noise))
    model.add(Dense(neurons, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=[r2_keras])
    return model

def build_deep_model_fn():
    model = Sequential()
    model.add(InputLayer(input_shape=(train.shape[1],)))
    model.add(GaussianNoise(0.25))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=[r2_keras])
    return model

epochs = 75
plot_start = 15

# model = build_model_fn()
# history = model.fit(train,
#                     y_train,
#                     epochs=epochs,
#                     verbose=2,
#                     validation_split=.15)
#
# dataz = pd.DataFrame({ 'loss': history.history['loss'][plot_start:],
#                        'val_loss': history.history['val_loss'][plot_start:],
#                        'epochs': range(plot_start+1,epochs+1) })
# dataz.plot(x='epochs')
# plt.show()
#
# exit()




models = [build_model_fn() for i in range(0, 10)]

for i, model in enumerate(models):
    print 'Training NN %d' % i
    model.fit(train, y_train, epochs=75, verbose=0)

predictions = np.column_stack([
    model.predict(test).flatten() for model in models
])

y_test = np.mean(predictions, axis=1)

#y_test = models[0].predict(test).flatten()

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('mercedes_submissions/nn.csv', index=False)


# pipe = make_pipeline(PCA(n_components=125),
#                      KerasRegressor(build_fn=build_model_fn, epochs=50, verbose=0))

# model = KerasRegressor(build_fn=build_model_fn, epochs=75, verbose=0)

# results = cross_val_score(model, train, y_train, cv=5, scoring='r2')
# print("Score: %.4f (%.4f)" % (results.mean(), results.std()))

# gsc = GridSearchCV(
#     estimator=model,
#     param_grid={
#         #'pca__n_components': range(100, 251, 50)
#         'neurons': range(18,31,4),
#         'layers': range(1,3),
#         #'epochs': range(25,151,25),
#         #'noise': [x/20.0 for x in range(3, 7)],
#         #'optimizer': ('adam', 'nadam'),
#     },
#     #scoring='r2',
#     scoring='neg_mean_squared_error',
#     cv=5
# )
#
# grid_result = gsc.fit(train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, test_stdev, train_mean, train_stdev, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['std_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['std_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f (%f) // Test : %f (%f) with: %r" % (train_mean, train_stdev, test_mean, test_stdev, param))
