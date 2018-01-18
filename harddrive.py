import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import struct

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# https://www.kaggle.com/devinanzelmo/cleaning-and-predictions-harddrive-failure
# https://www.kaggle.com/jeffbenshetler/cleaning-and-predictions-harddrive-failure

hdd = pd.read_csv('harddrive.csv')

print hdd.shape

# Drop any constant-value columns
for i in hdd.columns:
    if len(hdd.loc[:,i].unique()) == 1:
        hdd.drop(i, axis=1, inplace=True)

# Drop the normalized columns..
hdd = hdd.select(lambda x: x[-10:] != 'normalized', axis=1)

print hdd.shape

X = hdd.drop(['date', 'serial_number', 'model', 'capacity_bytes', 'failure'], axis=1)[:100000]
y = hdd['failure'][:100000]

X.fillna(value=0, inplace=True)


# gsc = GridSearchCV(
#      estimator=DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=20),
#      param_grid={
#          'class_weight': [{0: 1, 1: x} for x in range(150, 251, 25)]
#      },
#      scoring='f1',
#      cv=5
# )
#
# grid_result = gsc.fit(X, y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# for test_mean, train_mean, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))

# results = cross_val_score(tree, X, y, scoring='f1')
# print("Score: %.4f (%.4f)" % (results.mean(), results.std()))


# tree = DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=20, class_weight={0: 1, 1: 175})
# tree.fit(X, y)
# pred = tree.predict(X)

lr = LogisticRegression(class_weight={0: 1, 1: 175})
lr.fit(X, y)
pred = lr.predict(X)

# rf = RandomForestClassifier(class_weight={0: 1, 1: 175})
# rf.fit(X, y)
# pred = rf.predict(X)

print classification_report(y, pred)

#for feat, imp in zip(X.columns, rf.feature_importances_):
for feat, imp in zip(X.columns, lr.coef_[0]):
    if imp > 0.0001:
        print "- %s  => %.3f" % (feat, imp)



one_drive = hdd[hdd['serial_number'] == 'S30114J3']

print one_drive.shape
#print one_drive.head()

# #one_drive['smart_4_raw'].plot()
# #one_drive['smart_9_raw'].plot()
# #one_drive['smart_192_raw'].plot()
# #one_drive['smart_193_raw'].plot()
# one_drive['smart_197_raw'].plot()
# one_drive['smart_198_raw'].plot()
# one_drive['failure'].plot()
# plt.show()
#
# exit()