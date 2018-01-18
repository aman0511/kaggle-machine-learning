import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA, FastICA
import xgboost as xgb
from copy import deepcopy

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR

train = pd.read_csv('mercedes_train.csv')
test = pd.read_csv('mercedes_test.csv')

extra = pd.read_csv('mercedes-extra.csv')
train_extra = extra.join(test, on='ID', how='inner', rsuffix='_bla')
train_extra.drop(['ID_bla'], axis=1, inplace=True)
train = pd.concat([train, train_extra])

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_clean = pd.get_dummies(df_all, drop_first=True)

train_clean = df_clean[:num_train].values
test_clean = df_clean[num_train:].values

# Label-encoding object columns
for c in df_all.columns:
    if df_all[c].dtype == 'object':
        df_all[c] = LabelEncoder().fit_transform(df_all[c].values)

n_comp = 12

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results = tsvd.fit_transform(df_all)

pca = PCA(n_components=n_comp, random_state=420)
pca_results = pca.fit_transform(df_all)

ica = FastICA(n_components=n_comp, random_state=420)
ica_results = ica.fit_transform(df_all)

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results = grp.fit_transform(df_all)

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results = srp.fit_transform(df_all)

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    df_all['pca_' + str(i)] = pca_results[:, i - 1]
    df_all['ica_' + str(i)] = ica_results[:, i - 1]
    df_all['tsvd_' + str(i)] = tsvd_results[:, i - 1]
    df_all['grp_' + str(i)] = grp_results[:, i - 1]
    df_all['srp_' + str(i)] = srp_results[:, i - 1]

train = df_all[:num_train]
test = df_all[num_train:]

# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

#
# Create base models..
#

class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, np.log1p(y))

    def predict(self, X):
        return np.expm1(super(LogExpPipeline, self).predict(X))


svm_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                            SelectFromModel(Lasso(alpha=0.03)),
                                            SVR(kernel='rbf', C=0.5, epsilon=0.05)]))

en_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                           SelectFromModel(Lasso(alpha=0.03)),
                                           ElasticNet(alpha=0.001, l1_ratio=0.1)]))

rf_model = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                                 min_samples_leaf=25, max_depth=3)

et_model = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35,
                               max_features=150)

# keras_num_feats = 89
#
# def model_fn():
#     model = Sequential()
#     model.add(InputLayer(input_shape=(keras_num_feats,)))
#     model.add(GaussianNoise(0.25))
#     model.add(Dense(20, activation='tanh'))
#     model.add(Dense(1, activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer='nadam')
#     return model
#
# keras_pipe = make_pipeline(RobustScaler(),
#                            SelectFromModel(Lasso(alpha=0.03)),
#                            KerasRegressor(build_fn=model_fn, epochs=75, verbose=0))

#
# Train base & meta models
#

base_models = (rf_model, et_model, en_pipe, svm_pipe)

kfold = KFold(n_splits=5, shuffle=True)

meta_features = np.zeros((train_clean.shape[0], len(base_models)))

for i, regr in enumerate(base_models):
    print "Training base model #%d" % i
    predictions = np.zeros(train_clean.shape[0])
    for train_idx, holdout_idx in kfold.split(train_clean, y_train):
        instance = deepcopy(regr)
        instance.fit(train_clean[train_idx], y_train[train_idx])
        y_pred = instance.predict(train_clean[holdout_idx])
        meta_features[holdout_idx, i] = y_pred


xgb_params = {
    'eta': 0.0035,
    'max_depth': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,
    'silent': 1
}


#
# Train meta-xgboost on the out-of-fold predictions + original features
#
train_and_meta = np.hstack((train, meta_features))
dtrain = xgb.DMatrix(train_and_meta, y_train)

# Retrain base models on all data
for regr in base_models:
    regr.fit(train_clean, y_train)

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1500, early_stopping_rounds=50,
#                    verbose_eval=50, show_stdv=False)
# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# plt.show()
# print len(cv_output)
# exit()


model = xgb.train(xgb_params, dtrain, num_boost_round=1000)

#
# Now make predictions with base and meta model
#
meta_features = np.column_stack([ regr.predict(test_clean) for regr in base_models ])

dtest = xgb.DMatrix(np.hstack((test, meta_features)))

y_test = model.predict(dtest)

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('mercedes_submissions/xgb-stack.csv', index=False)


# print "Parameter search for meta model..."
# model = xgb.sklearn.XGBRegressor(max_depth=3, learning_rate=0.0035, subsample=0.9,
#                                 base_score=y_mean, objective='reg:linear',
#                                 n_estimators=850, colsample_bytree=0.8)
#
# gsc = GridSearchCV(
#     estimator=model,
#     param_grid={
#         #'learning_rate': (0.0025, 0.0030, 0.0035, 0.0040, 0.0045),
#         #'n_estimators': range(550, 951, 100),
#         #'max_depth': range(3,5),
#         #'subsample': (0.8, 0.9, 1.0),
#         #'colsample_bytree': (0.5, 0.6, 0.7, 0.8, 0.9),
#     },
#     scoring='r2',
#     cv=5
# )
#
# grid_result = gsc.fit(train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, train_mean, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))

