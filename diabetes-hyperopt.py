from sklearn import datasets
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from hyperopt import hp, tpe
from hyperopt.fmin import fmin

X,y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

def objective(space):
    clf = SVR(C=space['C'], epsilon=space['epsilon'])
    #return -1.0 * cross_val_score(clf, X, y, scoring='neg_mean_squared_error', cv=5).mean()
    return cross_val_score(clf, X, y, scoring='neg_mean_squared_error', cv=5).mean()

space = {
    'C': hp.quniform('C', 10.0, 5000.0, 10.0),
    'epsilon': hp.quniform('epsilon', 0.1, 10.0, 0.5),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=5,
            verbose=3)

print best
