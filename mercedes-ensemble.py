import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoLars, Lasso
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

train = df_all[:num_train]
test = df_all[num_train:]


class LogExpPipeline(Pipeline):
    def fit(self, X, y):
        super(LogExpPipeline, self).fit(X, np.log1p(y))

    def predict(self, X):
        return np.expm1(super(LogExpPipeline, self).predict(X))


svm_pipe = LogExpPipeline(_name_estimators([RobustScaler
                                            (),
                                            SelectFromModel(Lasso(alpha=0.03)),
                                            SVR(kernel='rbf', C=0.5, epsilon=0.05)]))

en_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
                                           SelectFromModel(Lasso(alpha=0.03)),
                                           ElasticNet(alpha=0.001, l1_ratio=0.1)]))

# keras_num_feats = 175
#
# def model_fn():
#     model = Sequential()
#     model.add(InputLayer(input_shape=(keras_num_feats,)))
#     model.add(GaussianNoise(0.2))
#     model.add(Dense(8, activation='relu'))
#     model.add(Dense(6, activation='relu'))
#     model.add(Dense(1, kernel_regularizer='l1_l2'))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
#
# keras_pipe = LogExpPipeline(_name_estimators([RobustScaler(),
#                                               PCA(n_components=keras_num_feats),
#                                               KerasRegressor(build_fn=model_fn, epochs=75, verbose=0)]))



from sklearn.base import BaseEstimator, TransformerMixin

class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self, transform_=None):
        self.transform_ = transform_

    def fit(self, X, y=None):
        self.transform_.fit(X, y)
        return self

    def transform(self, X, y=None):
        xform_data = self.transform_.transform(X, y)
        return np.append(X, xform_data, axis=1)



# results = cross_val_score(svm_pipe, train, y_train, cv=5, scoring='r2')
# print("SVM score: %.4f (%.4f)" % (results.mean(), results.std()))
# exit()

#
# XGBoost model
#

xgb_model = xgb.sklearn.XGBRegressor(max_depth=4, learning_rate=0.005, subsample=0.9, base_score=y_mean,
                                     objective='reg:linear', n_estimators=1300)


xgb_pipe = Pipeline(_name_estimators([AddColumns(transform_=PCA(n_components=9)),
                                      AddColumns(transform_=FastICA(n_components=9, max_iter=750, tol=0.001)),
                                      xgb_model]))


# results = cross_val_score(xgb_model, train, y_train, cv=5, scoring='r2')
# print("XGB score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Random Forest
#

rf_model = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                                 min_samples_leaf=25, max_depth=3)

# results = cross_val_score(rf_model, train, y_train, cv=5, scoring='r2')
# print("RF score: %.4f (%.4f)" % (results.mean(), results.std()))


#
# Extra Trees
#

#et_model = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35,
#                               max_features=150)

# results = cross_val_score(et_model, train, y_train, cv=5, scoring='r2')
# print("ET score: %.4f (%.4f)" % (results.mean(), results.std()))



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]

                print "Model %d fold %d score %f" % (i, j, r2_score(y_holdout, y_pred))

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
            S_test[:, i] = S_test_i.mean(axis=1)

        # results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        # print("Stacker score: %.4f (%.4f)" % (results.mean(), results.std()))
        # exit()

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)[:]
        return res

stack = Ensemble(n_splits=5,
                 #stacker=ElasticNetCV(l1_ratio=[x/10.0 for x in range(1,10)]),
                 stacker=ElasticNet(l1_ratio=0.1, alpha=1.4),
                 base_models=(svm_pipe, en_pipe, xgb_pipe, rf_model))

y_test = stack.fit_predict(train, y_train, test)

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('mercedes-submission.csv', index=False)
