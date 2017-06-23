from mlxtend.regressor import StackingRegressor
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

X, y = load_boston(return_X_y=True)

svr = SVR(kernel='linear')
lasso = Lasso()
rf = RandomForestRegressor(n_estimators=5)

stack = StackingRegressor(regressors=(svr, lasso, rf),
                          meta_regressor=lasso)

print('3-fold cross validation scores:\n')

for clf, label in zip([svr, lasso, rf, stack],
                      ['SVM', 'Lasso', 'Random Forest', 'StackingClassifier']):
    scores = cross_val_score(clf, X, y, cv=3, scoring='r2')
    print("R2 score: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))