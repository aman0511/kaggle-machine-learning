from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import numpy as np

X,y = datasets.load_diabetes(return_X_y=True)

gsc = GridSearchCV(
    estimator=SVR(C=1500, epsilon=7),
    param_grid={
        'C': range(500, 5001, 500),
        'epsilon': range(1, 10),
    },
    cv=25
)

grid_result = gsc.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, train_mean, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))