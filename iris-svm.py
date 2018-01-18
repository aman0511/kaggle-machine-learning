from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

X,y = datasets.load_iris(return_X_y=True)

model = SVC(C=1.5, gamma='auto')

gsc = GridSearchCV(
    estimator=model,
    param_grid={
        'C': np.linspace(0.5, 5.0),
        #'gamma': np.linspace(0.05, 0.5, 10),
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