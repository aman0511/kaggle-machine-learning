from sklearn import datasets
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost as xgb

X,y = datasets.load_diabetes(return_X_y=True)

gsc = GridSearchCV(
    estimator=xgb.sklearn.XGBRegressor(n_estimators=75, colsample_bytree=0.8, max_depth=2, subsample=0.5),
    param_grid={
        #'n_estimators': range(50,201,25),
        #'subsample': np.linspace(0.3, 1.0, 10),
        #'colsample_bytree': np.linspace(0.3, 1.0, 10),
        #'max_depth': range(2,6),
    },
    cv=5
)
grid_result = gsc.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, train_mean, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))