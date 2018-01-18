from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from pysmac.optimize import fmin

X,y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

def objective(x):
    clf = SVR(C=x[0], epsilon=x[1])
    return -1.0 * cross_val_score(clf, X, y, scoring='neg_mean_squared_error', cv=5).mean()

parameters, score = fmin(objective=objective,
                         x0=[10.0, 1.0],
                         xmin=[1.0, 0.001],
                         xmax=[10000.0, 10.0],
                         max_evaluations=25)

print('Lowest function value found: %f' % score)
print('Parameter setting %s' % parameters)

## http://pysmac.readthedocs.io/en/latest/quickstart.html