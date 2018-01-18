from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import tpot

X,y = datasets.load_diabetes(return_X_y=True)

config_dict = {

    # 'sklearn.linear_model.ElasticNet': {
    #     'l1_ratio': np.arange(0.05, 1.01, 0.05),
    #     'alpha': np.linspace(0.001, 10.0, 100),
    #     'normalize': [True, False]
    # },

    # 'sklearn.ensemble.ExtraTreesRegressor': {
    #     'n_estimators': range(50,501,50),
    #     'max_features': np.arange(0.05, 1.01, 0.05),
    #     'min_samples_split': range(2, 21),
    #     'min_samples_leaf': range(1, 21),
    #     'bootstrap': [True, False]
    # },
    #
    # 'sklearn.ensemble.GradientBoostingRegressor': {
    #     'n_estimators': [100],
    #     'loss': ["ls", "lad", "huber", "quantile"],
    #     'learning_rate': [1e-3, 1e-2, 1e-1, 0.5],
    #     'max_depth': range(2, 6),
    #     'min_samples_split': range(5, 26),
    #     'min_samples_leaf': range(5, 26),
    #     'subsample': np.arange(0.05, 1.01, 0.05),
    #     'max_features': np.arange(0.05, 1.01, 0.05),
    #     'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    # },

    # 'lightgbm.LGBMRegressor': {
    #     'boosting_type': ['gbdt', 'dart'],
    #     'objective': ['regression'],
    #     'drop_rate': [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
    #     'n_estimators': [250],
    #     'learning_rate': [0.005, 0.010, 0.015, 0.02, 0.025, 0.030],
    #     'num_leaves': range(2, 64, 2),
    #     'min_data_in_leaf': range(10,50,5),
    #     'colsample_bytree': np.linspace(0.1, 1.0, 20),
    #     'min_gain_to_split': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007],
    #     'subsample': [0.9, 0.95],
    #     'subsample_freq': [0, 1]
    # },

    # 'sklearn.ensemble.AdaBoostRegressor': {
    #     'n_estimators': [100],
    #     'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
    #     'loss': ["linear", "square", "exponential"],
    # },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    # 'sklearn.linear_model.Lasso': {
    #     'alpha': np.linspace(0.001, 1.0, 100),
    #     'normalize': [True, False]
    # },
    #
    # 'sklearn.linear_model.Ridge': {
    #     'alpha': np.linspace(0.5, 100.0, 100),
    #     'normalize': [True, False]
    # },

    # 'sklearn.ensemble.RandomForestRegressor': {
    #     'n_estimators': [100],
    #     'max_features': np.arange(0.25, 1.01, 0.05),
    #     'min_samples_split': range(4, 26),
    #     'min_samples_leaf': range(4, 26),
    #     'bootstrap': [True, False],
    #     'max_depth': range(2, 12),
    # },

    # 'sklearn.decomposition.PCA': {
    #     #'n_components': np.arange(0.5, 1.01, 0.01)
    #     'svd_solver': ['randomized'],
    #     'n_components': range(6,11)
    # },
    #
    # 'sklearn.decomposition.FastICA': {
    #     #'n_components': np.arange(0.5, 1.01, 0.01)
    #     'n_components': range(6,11)
    # },

    'sklearn.preprocessing.RobustScaler': {
    },

    # 'sklearn.preprocessing.StandardScaler': {
    # },

    # 'sklearn.feature_selection.VarianceThreshold': {
    #     'threshold': np.arange(0.05, 1.01, 0.05)
    # },
    #
    # 'sklearn.feature_selection.SelectFromModel': {
    #     'estimator': {
    #         'sklearn.linear_model.Lasso': {
    #             'alpha': np.linspace(0.001, 1.0, 100),
    #             'normalize': [True, False]
    #         }
    #     }
    # },
}

#y=np.log1p(y)

model = tpot.TPOTRegressor(generations=50, population_size=50, verbosity=3, config_dict=config_dict)
model.fit(X, y)

model.export('diabetes-tpot-result.py')

pipe = model._toolbox.compile(expr=model._optimized_pipeline)
cv_pred = cross_val_predict(pipe, X, y, cv=5)
print("R2 score: %.4f" % r2_score(y, cv_pred))