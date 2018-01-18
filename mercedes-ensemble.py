import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, InputLayer, GaussianNoise
from keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgbm


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

train = df_all[:num_train].values
test = df_all[num_train:].values


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
                                     objective='reg:linear', n_estimators=1000, colsample_bytree=0.7)


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
# LightGBM
#

lgbm_model = lgbm.LGBMRegressor(
    objective='regression',
    n_estimators=350,
    learning_rate=0.015,
    num_leaves=5,
    min_data_in_leaf=40,
    colsample_bytree=0.4,
    min_gain_to_split=0.0004,
    subsample=0.95,
    subsample_freq=1
)

#
# Extra Trees
#

et_model = ExtraTreesRegressor(n_estimators=100, n_jobs=4, min_samples_split=25, min_samples_leaf=35,
                               max_features=150)

# results = cross_val_score(et_model, train, y_train, cv=5, scoring='r2')
# print("ET score: %.4f (%.4f)" % (results.mean(), results.std()))


stack = StackingCVRegressor(#meta_regressor=Ridge(alpha=10),
                            meta_regressor=ElasticNet(l1_ratio=0.1, alpha=1.5),
                            regressors=(svm_pipe, en_pipe, xgb_pipe, rf_model, lgbm_model))
                            #regressors=(svm_pipe, en_pipe, xgb_pipe, rf_model))


# cv_pred = cross_val_predict(stack, train, y_train, cv=5)
# print("R2 score: %.4f" % r2_score(y_train, cv_pred))
# exit()

## R2 score: 0.5600 (en_pipe, rf_model)
## R2 score: 0.5601 (svm_pipe, en_pipe, xgb_pipe, rf_model, et_model)
## R2 score: 0.5605 (svm_pipe, en_pipe, xgb_pipe, rf_model, et_model, lgbm_model)
## R2 score: 0.5618 (svm_pipe, en_pipe, xgb_pipe, rf_model, lgbm_model)

stack.fit(train, y_train)

y_test = stack.predict(test)

df_sub = pd.DataFrame({'ID': id_test, 'y': y_test})
df_sub.to_csv('mercedes_submissions/ensemble.csv', index=False)


##
## https://www.kaggle.com/eaturner/stacking-em-up/output
##