import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import xgboost as xgb
from mlxtend.classifier import StackingClassifier, EnsembleVoteClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('titanic-train.csv')


def preprocess(data):
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

    data['Embarked'] = data['Embarked'].fillna('S')

    titles = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess":"Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty",
    }

    data['Title'] = data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip()).map(titles).astype('category')

    data['Age_Null'] = data['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    #data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['Age'] = data.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))

    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['FamilySize'] = data['FamilySize'].fillna(0)

    def fam_sz_class(size):
        if size == 0:
            return 'Solo'
        if size <= 2:
            return 'Normal'
        if size == 3:
            return 'Big'
        return 'Huge'

    data['FamilyClass'] = data['FamilySize'].map(fam_sz_class).astype('category')

    data['Name_Len'] = data['Name'].apply(lambda x: len(x))

    data['Ticket_Len'] = data['Ticket'].apply(lambda x: len(x))

    data['Cabin_Letter'] = data['Cabin'].apply(lambda x: str(x)[0].upper())

    data['Cabin_Num'] = data['Cabin']\
        .apply(lambda x: str(x).split(' ')[-1][1:])\
        .replace('an', np.NaN)\
        .apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
    data['Cabin_Num'] = pd.qcut(data['Cabin_Num'], 3, ['High', 'Medium', 'Low'])

    data['Ticket_Lett'] = data['Ticket'].apply(lambda x: str(x)[0]).apply(lambda x: str(x))
    data['Ticket_Lett'] = np.where((data['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), data['Ticket_Lett'],
                                   np.where((data['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), 'LOW', 'OTHER'))

    # Do the one-hot encoding
    data = pd.get_dummies(data, columns=['Sex', 'Title', 'Pclass', 'Embarked', 'FamilyClass', 'Cabin_Letter', 'Cabin_Num', 'Ticket_Lett'])

    return data

data = preprocess(data)

wanted_cols = ['Pclass_1', 'Pclass_2', 'Pclass_3',
               'Sex_male', 'Sex_female',
               'Name_Len',
               'Age', 'Age_Null', #'Parch', 'SibSp',
               'Fare',
               'Ticket_Len',
               'Ticket_Lett_1', 'Ticket_Lett_2', 'Ticket_Lett_3', 'Ticket_Lett_S', 'Ticket_Lett_P', 'Ticket_Lett_C', 'Ticket_Lett_A',
               'Ticket_Lett_LOW', 'Ticket_Lett_OTHER',
               'Title_Officer', 'Title_Royalty', 'Title_Master',
               'Title_Mr', 'Title_Mrs', 'Title_Miss',
               'Embarked_C', 'Embarked_Q', 'Embarked_S',
               'Cabin_Letter_A', 'Cabin_Letter_B', 'Cabin_Letter_C', 'Cabin_Letter_D', 'Cabin_Letter_E',
               'Cabin_Letter_F', 'Cabin_Letter_G', 'Cabin_Letter_N',
               'Cabin_Num_Low', 'Cabin_Num_Medium', 'Cabin_Num_High',
               #'FamilySize',
               'FamilyClass_Solo', 'FamilyClass_Normal', 'FamilyClass_Big', 'FamilyClass_Huge']

def getX(data):
    return data.as_matrix(wanted_cols)


# gsc = GridSearchCV(
#     estimator=xgb.XGBClassifier(n_estimators=150, max_depth=4, subsample=0.7, colsample_bytree=0.4),
#     param_grid={
#         #'n_estimators': range(10,201,10),
#         #'learning_rate': (0.02, 0.04, 0.06, 0.08, 0.10),
#         #'subsample': (0.5, 0.6, 0.7, 0.8, 0.9),
#         #'colsample_bytree': (0.3, 0.4, 0.5, 0.6, 0.7),
#         #'max_depth': range(2,7),
#         #'reg_alpha': (0.0, 0.1, 0.2, 0.3),
#         #'reg_lambda': (0.8, 0.9, 1.0),
#         #'gamma': (0, 0.005, 0.01),
#     },
#     cv=5
# )
#
# grid_result = gsc.fit(getX(data), data['Survived'])
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

classifier = EnsembleVoteClassifier(
    clfs=(
        xgb.XGBClassifier(n_estimators=150, max_depth=4, subsample=0.7, colsample_bytree=0.4),
        RandomForestClassifier(n_estimators=500, max_features='auto', min_samples_split=20, min_samples_leaf=5),
        GradientBoostingClassifier(n_estimators=250, max_features=5, min_samples_leaf=5),
        ExtraTreesClassifier(n_estimators=75, min_samples_leaf=5)
    )
)

# classifier = StackingClassifier(
#     classifiers=(
#         xgb.XGBClassifier(n_estimators=150, max_depth=4, subsample=0.7, colsample_bytree=0.4),
#         RandomForestClassifier(n_estimators=500, max_features='auto', min_samples_split=20, min_samples_leaf=5),
#         GradientBoostingClassifier(n_estimators=250, max_features=5, min_samples_leaf=5),
#         ExtraTreesClassifier(n_estimators=75, min_samples_leaf=5)
#     ),
#     meta_classifier=LogisticRegression()
# )

# scores = cross_val_score(voting_classifier, getX(data), data['Survived'], cv=5, scoring='accuracy')
# print np.mean(scores)

classifier.fit(getX(data), data['Survived'])

data = pd.read_csv('titanic-test.csv')
data = preprocess(data)

data['Survived'] = classifier.predict(getX(data))

data.to_csv('titanic-submission.csv', index=False, columns=['PassengerId', 'Survived'])