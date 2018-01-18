from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

X,y = datasets.load_iris(return_X_y=True)

model = GradientBoostingClassifier(
    n_estimators=10,
    learning_rate=0.1,
    max_depth=3,
    max_features=0.6,
    min_samples_leaf=5,
    min_samples_split=5,
    subsample=1.0
)

gsc = GridSearchCV(
    estimator=model,
    param_grid={
        #'n_estimators': [10],
        #'learning_rate': [0.1],
        #'max_depth': range(2,7),
        #'max_features': np.linspace(0.1, 1.0),
        #'min_samples_leaf': range(2,30,3),
        #'min_samples_split': range(2,30,3),
        #'subsample': np.linspace(0.1, 1.0, 10),
    },
    scoring='f1_micro',
    cv=5
)

grid_result = gsc.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, train_mean, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))