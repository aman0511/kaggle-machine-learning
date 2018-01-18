import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from imblearn.over_sampling import SMOTE

# Read data
data = pd.read_csv("creditcard.csv",header = 0)
data.drop('Time', axis=1, inplace=True)

# Separate X/y from data
y = data['Class'].values
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


# gsc = GridSearchCV(
#     estimator=LogisticRegression(C=0.05, class_weight={0: 1, 1: 6}),
#     param_grid={
#         #'C': (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
#         'class_weight': [{0: 1, 1: x} for x in range(4,9)]
#     },
#     scoring='f1',
#     #scoring='roc_auc',
#     cv=5
# )
# grid_result = gsc.fit(X, y)

## Best: 0.801641 using {'C': 0.3, 'class_weight': {0: 0.14, 1: 0.86}}
## Best: 0.801463 using {'C': 0.005, 'class_weight': {0: 1, 1: 6}}

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, train_mean, param in zip(
#         grid_result.cv_results_['mean_test_score'],
#         grid_result.cv_results_['mean_train_score'],
#         grid_result.cv_results_['params']):
#     print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))
#
# exit()


# Fit the model using best params
# lr = LogisticRegression(**grid_result.best_params_)
lr = LogisticRegression(C=0.05, class_weight={0: 1, 1: 6})
lr.fit(X_train, y_train)

# Predict..
y_pred = lr.predict(X_test)
predict_proba = lr.predict_proba(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

#print roc_auc_score(y_test, predict_proba[:,1])


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





from matplotlib import pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
