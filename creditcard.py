import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from matplotlib import pyplot as plt

# Read data
data = pd.read_csv("creditcard.csv",header = 0)
data.drop('Time', axis=1, inplace=True)

# Separate X/y from data
y = data['Class'].values
#y = -y + 1
X = data.drop('Class', axis=1).values

# Scaling..
scaler = RobustScaler()
X = scaler.fit_transform(X)

# Split into train/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


##
## This data has very imbalanced classes, which we need to deal
## with to get good results. There is multile approaches:
## - resampling to balance clasess (under/over/...)
## - assign class weights on the classifier
##
# from imblearn.over_sampling import SMOTE
# smote = SMOTE()
# X_train_r, y_train_r = smote.fit_sample(X_train, y_train)


gsc = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={
        'C': (0.005, 0.01, 0.05),
        'class_weight': [{0: 1, 1: x} for x in range(6,10)]
    },
    scoring='f1',
    cv=5
)

grid_result = gsc.fit(X, y)

## Best: 0.801641 using {'C': 0.3, 'class_weight': {0: 0.14, 1: 0.86}}

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, train_mean, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))




# Fit the model using best params
lr = LogisticRegression(**grid_result.best_params_)
lr.fit(X_train, y_train)

# Predict..
y_pred = lr.predict(X_test)


# Evaluate the model
print classification_report(y_test, y_pred)


##
## Precision vs recall trade-off
## False-positives vs true positives tradeoff
##
# predict_proba = lr.predict_proba(X_test)
# precision, recall, _ = precision_recall_curve(y_test, predict_proba[:,1])
# fpr, tpr, _ = roc_curve(y_test, predict_proba[:,1])
#
# # Plot 1
# plt.clf()
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall curve')
# plt.show()
#
# # Plot 2
# plt.clf()
# plt.plot(fpr, tpr)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC curve')
# plt.show()

