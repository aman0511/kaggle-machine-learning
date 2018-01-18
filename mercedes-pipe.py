import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVR
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet, Lasso, LassoCV, Ridge, RidgeCV,\
    LinearRegression, LassoLars, LassoLarsCV, LassoLarsIC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

train = pd.read_csv('mercedes_train.csv')
test = pd.read_csv('mercedes_test.csv')

extra = pd.read_csv('mercedes-extra.csv')
train_extra = extra.join(test, on='ID', how='inner', rsuffix='_bla')
train_extra.drop(['ID_bla'], axis=1, inplace=True)
train = pd.concat([train, train_extra])

y_train = np.log1p(train['y'].values)
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

# Label-encoding object columns
# for c in df_all.columns:
#     if df_all[c].dtype == 'object':
#         df_all[c] = LabelEncoder().fit_transform(df_all[c].values)

train = df_all[:num_train]
test = df_all[num_train:]

# pipe = make_pipeline(RobustScaler(),
#                      SelectFromModel(Lasso(alpha=0.001)),
#                      SVR(kernel='rbf', C=1.0, epsilon=0.05))
#
# pipe.fit(train, y_train)
# df_sub = pd.DataFrame({'ID': id_test, 'y': np.expm1(pipe.predict(test))})
# df_sub.to_csv('mercedes_submissions/svr.csv', index=False)
#
#
# pipe = make_pipeline(RobustScaler(),
#                      SelectFromModel(Lasso(alpha=0.001)),
#                      ElasticNet(alpha=0.001, l1_ratio=0.1))
#
# pipe.fit(train, y_train)
# df_sub = pd.DataFrame({'ID': id_test, 'y': np.expm1(pipe.predict(test))})
# df_sub.to_csv('mercedes_submissions/elasticnet.csv', index=False)
#
# exit()


pipe = make_pipeline(RobustScaler(),
                     #PCA(n_components=150),
                     #PCA(),
                     #SelectFromModel(Ridge(alpha=35)),
                     SelectFromModel(Lasso(alpha=0.001)),
                     #SelectFromModel(LassoCV()),
                     #ElasticNet(alpha=0.001, l1_ratio=0.1))
                     Ridge(alpha=10.0))
                     #Lasso(alpha=0.001))
                     #SVR(kernel='rbf', C=1.0, epsilon=0.05)) # With PCA
                     #SVR(kernel='rbf', C=0.5, epsilon=0.05)) # With SelectFromModel

# pipe = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
#                              min_samples_leaf=25, max_depth=3)

# results = cross_val_score(pipe, train, y_train, cv=10, scoring='r2')
# print("Score: %.4f (%.4f)" % (results.mean(), results.std()))
# exit()

pipe.fit(train, y_train)

cv_pred = cross_val_predict(pipe, train, y_train, cv=5)
print("R2 score: %.4f" % r2_score(y_train, cv_pred))
exit()


## R2 score: 0.6101 => SelectFromModel + SVR(C=1)
## R2 score: 0.6101 => SelectFromModel + SVR(C=0.5)
## R2 score: 0.6091 => SelectFromModel + SVR(C=0.5) + LabelEncoder
## R2 score: 0.5919 => SelectFromModel + ElasticNet
## R2 score: 0.5994 => SelectFromModel + Ridge(alpha=10)
## R2 score: 0.5941 => PCA(150) + ElasticNet
## R2 score: 0.6082 => PCA + SVR
## R2 score: 0.6082 => SVR
## R2 score: 0.5895 => Lasso(alpha=0.001)
## R2 score: 0.6092 => RandomForestRegressor


# gsc = GridSearchCV(
#     estimator=pipe,
#     param_grid={
#         #'pca__n_components': (0.95, 0.975, 0.99, None),
#         #'pca__n_components': range(75,400,25),
#         'svr__C': (0.1, 0.25, 0.5, 1.0),
#         #'svr__epsilon': (0.01, 0.05, 0.1),
#         #'elasticnet__alpha': (0.0005, 0.001, 0.005),
#         #'elasticnet__l1_ratio': (0.05, 0.1, 0.2),
#         #'sequentialfeatureselector__k_features': range(100,151,25)
#     },
#     scoring='r2',
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
#
# pipe.set_params(**grid_result.best_params_)


pipe.fit(train, y_train)

y_test = np.expm1(pipe.predict(test))

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('mercedes-submission.csv', index=False)


##
## Apply ideas from:
## -> https://www.kaggle.com/puremath86/easy-feature-selection-pipeline-0-55-at-lb
##
