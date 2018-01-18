from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit, \
    cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from scipy.stats.distributions import uniform, randint

import pandas as pd

df_partial = pd.read_csv('porto-train.csv', nrows=50000)
df_train = pd.read_csv('porto-train.csv')
df_test = pd.read_csv('porto-test.csv')
df_sample = pd.read_csv('porto-sample_submission.csv')

X = df_train.drop(['id', 'target'], axis=1)
Y = df_train['target']

X_partial = df_partial.drop(['id', 'target'], axis=1)
Y_partial = df_partial['target']

rf = RandomForestClassifier(n_estimators=1000, n_jobs=4, class_weight='balanced',
                            min_samples_leaf=25, min_samples_split=25)

# rf.fit(X, Y)
#
# y_pred = rf.predict_proba(df_test.drop('id', axis=1))
# y_pred = y_pred[:,1]
#
# df_sample['target'] = y_pred
# df_sample.to_csv('porto-submission-rf.csv', index=False)
#
# exit()

def gini(solution, submission):
    df = zip(solution, submission, range(len(solution)))
    df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

gini_scorer = make_scorer(normalized_gini, greater_is_better=True)

cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2)

gsc = GridSearchCV(
    estimator=rf,
    param_grid={
        #'class_weight': [{0: 1, 1: x} for x in range(300, 701, 100)] + ['balanced'],
        #'min_samples_leaf': range(5,51,5),
        #'min_samples_split': range(5,56,10),
        #'n_estimators': range(200, 601, 200),
        #'criterion': ('gini', 'entropy')
        #'max_features': ('auto', 'sqrt'),
        #'max_features': range(3, 9),
        #'max_depth': range(3, 7),
    },
    #scoring='neg_log_loss',
    scoring='roc_auc',
    #scoring='f1',
    #scoring=gini_scorer,
    cv=cv,
    verbose=2
)

rsc = RandomizedSearchCV(
     estimator=rf,
     param_distributions={
         'n_estimators': randint(250, 2500),
         #'class_weight': [{0: 1, 1: x} for x in range(15, 51, 5)],
         #'criterion': ('gini', 'entropy'),
         #'min_samples_leaf': randint(15, 50),
         #'min_samples_split': randint(15, 50),
         #'max_features': ('auto', 'sqrt') + range(5,50),
         #'max_features': range(3, 9),
         #'max_depth': randint(2, 6),
     },
     #scoring=gini_scorer,
     scoring='roc_auc',
     cv=cv,
     verbose=2,
     n_iter=7
)

# grid_result = rsc.fit(X, Y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, train_mean, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))


from hyperopt import hp, tpe
from hyperopt.fmin import fmin


def objective(space):
    clf = RandomForestClassifier(n_estimators=750, n_jobs=4, class_weight='balanced',
                                 min_samples_leaf=space['min_samples_leaf'],
                                 min_samples_split=space['min_samples_split'])

    score = cross_val_score(clf, X_partial, Y_partial, scoring='roc_auc', cv=cv).mean()

    print("min_samples_leaf = {} ; min_samples_split = {} => {}".format(
        space['min_samples_leaf'], space['min_samples_split'], score
    ))

    return -1.0 * score

space = {
    'min_samples_leaf': 5 + hp.randint('min_samples_leaf', 100),
    'min_samples_split': 5 + hp.randint('min_samples_split', 100),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            verbose=3)

print best