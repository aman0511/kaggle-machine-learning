import numpy as np
from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l1, l2, l1_l2


df_train = pd.read_csv("sberbank_train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("sberbank_test.csv", parse_dates=['timestamp'])
#df_macro = pd.read_csv("sberbank_macro.csv", parse_dates=['timestamp'])

df_all = pd.concat([df_train, df_test])
df_all.drop(['id', 'price_doc'], axis=1, inplace=True)
#df_all.drop(['timestamp'], axis=1, inplace=True)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek
df_all.drop(['timestamp'], axis=1, inplace=True)

numeric_feats = df_all.dtypes[df_all.dtypes != "object"].index

num_train = len(df_train)
df_all = pd.get_dummies(df_all)

df_all = df_all.fillna(df_all.mean())

# Log transform skewed features
skewness = df_all[numeric_feats].apply(lambda x: skew(x.dropna()))
left_skewed_feats = skewness[skewness > 0.5].index
right_skewed_feats = skewness[skewness < -0.5].index
df_all[left_skewed_feats] = np.log1p(df_all[left_skewed_feats])
#all_data[right_skewed_feats] = np.exp(all_data[right_skewed_feats])

scaler = RobustScaler()
df_all[numeric_feats] = scaler.fit_transform(df_all[numeric_feats])

x_train = df_all[:num_train]
x_test = df_all[num_train:]

y_train = np.log1p(df_train['price_doc'].values)


###
##  Make predictions
###

def build_model(neurons=8):
    model = Sequential()
    model.add(InputLayer(input_shape=(x_train.shape[1],)))
    model.add(Dense(neurons, activation='relu', kernel_regularizer='l1_l2'))
    model.add(Dense(1, kernel_regularizer='l1_l2'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = build_model()
# history = model.fit(x_train.values,
#                     y_train,
#                     epochs=300,
#                     verbose=2,
#                     validation_split=.15)
history = model.fit(x_train.values,
                    y_train,
                    epochs=300,
                    verbose=2)

y_test = np.expm1(model.predict(x_test.values))

df_sub = pd.DataFrame({'id': df_test['id'], 'price_doc': [x[0] for x in y_test.tolist()]})
df_sub.to_csv('sberbank-submission-keras.csv', index=False)

# dataz = pd.DataFrame({ 'loss': history.history['loss'][2:],
#                        'val_loss': history.history['val_loss'][2:],
#                        'epochs': range(3,301) })
# dataz.plot(x='epochs')
# plt.show()

# estimator = KerasRegressor(build_fn=build_model, nb_epoch=10, verbose=2)
# results = cross_val_score(estimator, x_train.values, y_train, cv=5, scoring='neg_mean_squared_error')
# print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
# exit()

# kernel=l2, activity=l1         ==>> MSE = -0.3265
# kernel=l1_l2, activity=l1_l2   ==>> MSE = -15.3907
# kernel=l1_l2                   ==>> MSE = -0.2990
# kernel=l1_l2, bias=l1_l2       ==>> MSE = -0.31xx



# Layers (1,)       ==>> Results: 0.4801 (0.0960) MSE + dropout + epoch=100
# Layers (2,)       ==>> Results: 0.4011 (0.1082) MSE + dropout
# Layers (6,)       ==>> Results: 0.2643 (0.0845) MSE + epoch=100
# Layers (8,)       ==>> Results: 0.2578 (0.0929) MSE
# Layers (10,)      ==>> Results: 0.2880 (0.0492) MSE + epoch=100
# Layers (12,)      ==>> Results: 0.2621 (0.0952) MSE
# Layers (20,)      ==>> Results: 0.2559 (0.0836) MSE
# Layers (32,)      ==>> Results: 0.2590 (0.0800) MSE
# Layers (64,)      ==>> Results: 0.2739 (0.1015) MSE
# Layers (128,)     ==>> Results: 0.2713 (0.0917) MSE

# Layers (12,4,)    ==>> Results: -0.2626 (0.0501) MSE

# gsc = GridSearchCV(
#     estimator=KerasRegressor(build_fn=build_model, nb_epoch=10, verbose=0),
#     param_grid={
#         #'neurons': range(1,30,3),
#         #'dropout': (0.1, 0.2, 0.3, 0.4, 0.5)
#         'nb_epoch': range(1,30,5),
#     },
#     scoring='neg_mean_squared_error',
#     cv=5
# )
# grid_result = gsc.fit(x_train.values, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#
# dataz = pd.DataFrame({ 'test_score': grid_result.cv_results_['mean_test_score'],
#                        'train_score': grid_result.cv_results_['mean_train_score'],
#                        'nb_epoch': range(1,30,5) })
# dataz.plot(x='nb_epoch')
# plt.show()


#
# http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
#


# plt.figure()
# plt.title('Learning Curve')
# plt.xlabel("# Samples")
# plt.ylabel("Score")
# train_sizes = range(2000,20000,2000)
# train_sizes, train_scores, test_scores = learning_curve(
#     estimator, x_train.values, y_train, cv=5, train_sizes=train_sizes)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
# plt.grid()
#
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.1,
#                  color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#          label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#          label="Cross-validation score")
#
# plt.legend(loc="best")
# plt.show()