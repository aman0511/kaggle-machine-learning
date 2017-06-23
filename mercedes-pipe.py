import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Lasso, LassoCV, Ridge, RidgeCV,\
    LinearRegression, LassoLars, LassoLarsCV, LassoLarsIC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

train = pd.read_csv('mercedes_train.csv')
test = pd.read_csv('mercedes_test.csv')

y_train = np.log1p(train['y'].values)
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]

pipe = make_pipeline(RobustScaler(),
                     #PCA(n_components=125),
                     #PCA(),
                     #SelectFromModel(Ridge(alpha=35)),
                     #SelectFromModel(Lasso(alpha=0.03)),
                     SelectFromModel(LassoCV()),
                     ElasticNet(alpha=0.001, l1_ratio=0.1))
                     #LassoLars())
                     #SVR(kernel='rbf', C=1.0, epsilon=0.05)) # With PCA
                     #SVR(kernel='rbf', C=0.5, epsilon=0.05)) # With SelectFromModel

# pipe = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
#                              min_samples_leaf=25, max_depth=3)

results = cross_val_score(pipe, train, y_train, cv=10, scoring='r2')
print("Score: %.4f (%.4f)" % (results.mean(), results.std()))
exit()

## Score: 0.6216 (0.0664) => SelectFromModel + SVR
## Score: 0.6216 (0.0657) => SelectFromModel + ElasticNet
## Score: 0.6079 (0.0646) => PCA(125) + ElasticNet
## Score: 0.6200 (0.0685) => PCA + SVR
## Score: 0.6200 (0.0685) => SVR
## Score: 0.6192 (0.0702) => SelectFromModel + RandomForestRegressor
## Score: 0.6222 (0.0662) => RandomForestRegressor
## Score: 0.5876 (0.0589) => LassoLarsCV
## Score: 0.5927 (0.0593) => LassoLarsIC

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
