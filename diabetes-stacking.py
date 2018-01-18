import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from mlxtend.regressor import StackingCVRegressor
import xgboost as xgb

X,y = datasets.load_diabetes(return_X_y=True)

#X,y = datasets.load_boston(return_X_y=True)
#y = np.log1p(y)
#X = RobustScaler().fit_transform(X)

class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, regressors):
        self.regressors = regressors
        self.predictions = None

    def fit(self, X, y):
        for regr in self.regressors:
            regr.fit(X, y)
        return self

    def predict(self, X):
        self.predictions = np.column_stack([regr.predict(X) for regr in self.regressors])
        return np.mean(self.predictions, axis=1)

lasso = Lasso(alpha=0.01)
ridge = Ridge(alpha=0.1)
en = ElasticNet(alpha=0.0005, l1_ratio=0.5)
svr = SVR(C=1500, epsilon=7)
rf = RandomForestRegressor(n_estimators=75, min_samples_leaf=6, min_samples_split=2, max_depth=4, max_features=5)
gbm  = GradientBoostingRegressor(subsample=0.7, min_samples_leaf=6)
xgbm = xgb.sklearn.XGBRegressor(n_estimators=75, colsample_bytree=0.8, max_depth=2, subsample=0.5)
stacked = StackingCVRegressor(regressors=(lasso, ridge, en, xgbm, svr, rf), meta_regressor=Lasso(alpha=128), cv=5, use_features_in_secondary=True)
stack_nofeats = StackingCVRegressor(regressors=(lasso, ridge, en, xgbm, svr, rf), meta_regressor=Lasso(alpha=15), cv=5)
average = AveragingRegressor((lasso, ridge, en, svr, rf, gbm, xgbm, stacked))

# lasso.fit(X, y)
# ridge.fit(X, y)
# stack_nofeats.fit(X,y)
# stacked.fit(X,y)
# exit()

# import seaborn as sns
# import matplotlib.pyplot as plt
#
# average.fit(X,y)
# average.predict(X)
# sns.pairplot(pd.DataFrame(average.predictions))
# plt.show()
#
# exit()

models = [
    ('Lasso', lasso),
    ('Ridge', ridge),
    ('ElasticNet', en),
    ('SVR', svr),
    ('DecisionTreeRegressor', DecisionTreeRegressor(min_samples_leaf=25)),
    ('RandomForestRegressor', rf),
    ('GradientBoostingRegressor', gbm),
    ('xgboost', xgbm),
    ('Averaged', average),
    ('Stacked', stacked),
    ('StackedNoFeats', stack_nofeats),
]

for label, model in models:
    #scores = cross_val_score(model, X, y, cv=KFold(n_splits=25, shuffle=True), scoring='r2')
    #scores = cross_val_score(model, X, y, cv=ShuffleSplit(n_splits=25, test_size=0.15), scoring='r2')
    scores = cross_val_score(model, X, y, cv=ShuffleSplit(n_splits=10, test_size=0.15), scoring='neg_mean_squared_error')
    print("R2 score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# R2 score: 0.47 (+/- 0.08) [Lasso]
# R2 score: 0.49 (+/- 0.07) [Ridge]
# R2 score: 0.50 (+/- 0.06) [ElasticNet]
# R2 score: 0.50 (+/- 0.07) [SVR]
# R2 score: 0.35 (+/- 0.12) [DecisionTreeRegressor]
# R2 score: 0.44 (+/- 0.06) [RandomForestRegressor]
# R2 score: 0.44 (+/- 0.09) [xgboost]
# R2 score: 0.51 (+/- 0.06) [Averaged]
# R2 score: 0.48 (+/- 0.06) [Stacked]
# R2 score: 0.50 (+/- 0.07) [StackedNoFeats]

## R2 score: 0.5164  TPOT => RandomForestRegressor(ExtraTreesRegressor(ElasticNet(input_matrix, ElasticNet__alpha=1.617, ElasticNet__l1_ratio=0.45, ElasticNet__normalize=DEFAULT), ExtraTreesRegressor__bootstrap=DEFAULT, ExtraTreesRegressor__max_features=0.75, ExtraTreesRegressor__min_samples_leaf=8, ExtraTreesRegressor__min_samples_split=11, ExtraTreesRegressor__n_estimators=450), RandomForestRegressor__bootstrap=DEFAULT, RandomForestRegressor__max_depth=7, RandomForestRegressor__max_features=0.85, RandomForestRegressor__min_samples_leaf=16, RandomForestRegressor__min_samples_split=17, RandomForestRegressor__n_estimators=100)
