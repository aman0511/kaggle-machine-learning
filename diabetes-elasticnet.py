from sklearn import datasets
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.linear_model import ElasticNet
import numpy as np

X,y = datasets.load_diabetes(return_X_y=True)

gsc = GridSearchCV(
    estimator=ElasticNet(alpha=0.0005, l1_ratio=0.5),
    param_grid={
        'alpha': np.linspace(0.00001, 0.001),
        #'l1_ratio': np.linspace(0.1, 1.0, 9),
    },
    #cv=ShuffleSplit(n_splits=10, test_size=0.15)
    cv=KFold(n_splits=25)
)

grid_result = gsc.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, train_mean, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))