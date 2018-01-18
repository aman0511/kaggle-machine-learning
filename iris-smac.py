from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from pysmac.optimize import fmin

X,y = datasets.load_iris(return_X_y=True)

def objective(x):
    #clf = SVC(C=x[0])
    clf = LogisticRegression(C=x[0], solver='lbfgs', multi_class='multinomial')
    return -1.0 * cross_val_score(clf, X, y, cv=5).mean()

parameters, score = fmin(objective=objective,
                         x0=[1.0],
                         xmin=[0.001],
                         xmax=[100.0],
                         max_evaluations=10)

# def objective(x_int):
#     #clf = DecisionTreeClassifier(min_samples_leaf=x_int[0], min_samples_split=x_int[1])
#     clf = RandomForestClassifier(max_depth=x_int[0], min_samples_leaf=x_int[1], min_samples_split=x_int[2])
#     return -1.0 * cross_val_score(clf, X, y, cv=5).mean()
#
# parameters, score = fmin(objective=objective,
#                          x0_int=[2, 50, 50],
#                          xmin_int=[2, 1, 2],
#                          xmax_int=[5, 100, 100],
#                          max_evaluations=20)

print('Lowest function value found: %f' % score)
print('Parameter setting %s' % parameters)