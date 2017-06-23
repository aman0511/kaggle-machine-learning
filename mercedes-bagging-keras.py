import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoLars
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, InputLayer, GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb


train = pd.read_csv('mercedes_train.csv')
test = pd.read_csv('mercedes_test.csv')

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

df_all = RobustScaler().fit_transform(df_all)

train = df_all[:num_train]
test = df_all[num_train:]

model = BaggingRegressor(
    n_estimators=100,
    max_features=40
)

# results = cross_val_score(stack, train, y_train, cv=5, scoring='r2')
# print("Score: %.4f (%.4f)" % (results.mean(), results.std()))
# exit()
#
#
#
# y_test = stack.fit_predict(train, y_train, test)
#
# df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
# df_sub.to_csv('mercedes-submission.csv', index=False)


gsc = GridSearchCV(
    estimator=model,
    param_grid={
        'n_estimators': range(75,126,25),
        'max_features': range(75,126,25)
    },
    scoring='r2',
    cv=5
)

grid_result = gsc.fit(train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, train_mean, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))