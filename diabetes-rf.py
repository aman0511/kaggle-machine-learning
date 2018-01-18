from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

X,y = datasets.load_diabetes(return_X_y=True)

gsc = GridSearchCV(
    estimator=RandomForestRegressor(n_estimators=75,
                                    max_features=5,
                                    n_jobs=4,
                                    max_depth=4,
                                    min_samples_leaf=6),
    param_grid={
        'n_estimators': range(45,76,5)
        #'max_features': range(1,11),
        #'min_samples_leaf': range(2,13,2),
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