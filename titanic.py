import pandas as pd
from sklearn import linear_model, svm, tree, naive_bayes
from sklearn.model_selection import cross_val_score, GridSearchCV, validation_curve
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt



# https://www.kaggle.com/jerrytseng/titanic/titanic-random-forest-82-78


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

    #print data['Survived'].groupby(data['Cabin_Letter']).mean()
    #print
    #print data['Survived'].groupby(data['Cabin_Letter']).sum()
    #print

    data['Ticket_Lett'] = data['Ticket'].apply(lambda x: str(x)[0]).apply(lambda x: str(x))
    data['Ticket_Lett'] = np.where((data['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), data['Ticket_Lett'],
                                   np.where((data['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), 'LOW', 'OTHER'))

    # Do the one-hot encoding
    data = pd.get_dummies(data, columns=['Sex', 'Title', 'Pclass', 'Embarked', 'FamilyClass', 'Cabin_Letter', 'Cabin_Num', 'Ticket_Lett'])


    return data

data = preprocess(data)

#exit()


"""
print(data['Cabin_Letter_A'].astype('category').describe())
print
print(data['Survived'].describe())
print()
print(data['Age'].describe())
print()
"""

#print data['FamilySize'].describe()
#print
#print data['Survived'].groupby(data['FamilyClass']).mean()
#print

#exit()

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

classifiers = [
    ('Logistic regression', linear_model.LogisticRegression()),
    #('SVM classifier', svm.SVC(kernel='linear')),
    #('Decision tree classifier', tree.DecisionTreeClassifier()),
    #('Naive Bayes classifier', naive_bayes.GaussianNB()),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=250, max_features=5)),
    ('Random Forest', RandomForestClassifier(n_estimators=250, max_features='auto', min_samples_split=10, min_samples_leaf=1, oob_score=True)),
]

for name, candidate in classifiers:
    scores = cross_val_score(candidate, getX(data), data['Survived'], cv=5, scoring='accuracy')
    print(name, np.mean(scores))
exit()


model = RandomForestClassifier(n_estimators=500, max_features='auto', min_samples_split=20, min_samples_leaf=5)
#model = GradientBoostingClassifier(n_estimators=25, learning_rate=0.1, max_features='auto', min_samples_split=20, min_samples_leaf=5, max_depth=5)
model.fit(getX(data), data['Survived'])

# print pd.concat((pd.DataFrame(wanted_cols, columns = ['variable']),
#                  pd.DataFrame(model.feature_importances_, columns = ['importance'])),
#                 axis = 1).sort_values(by='importance', ascending = False)[:35]

data = pd.read_csv('titanic-test.csv')

data = preprocess(data)

data['Survived'] = model.predict(getX(data))

data.to_csv('titanic-submission.csv', index=False, columns=['PassengerId', 'Survived'])

"""
gsc = GridSearchCV(
    estimator=GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_features='auto', min_samples_split=20, min_samples_leaf=5, max_depth=6),
    param_grid={
        'n_estimators': range(10,100,20),
        'max_depth': range(1,10),
        'min_samples_split': range(5,20,5),
    },
    cv=5
)
grid_result = gsc.fit(getX(data), data['Survived'])

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
"""

"""
#param = 'min_samples_split'
#param_range = range(2,40,4)
#classifier = RandomForestClassifier(n_estimators=50, max_features='auto', min_samples_split=20, min_samples_leaf=5)

param = 'n_estimators'
#param_range = (0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15)
param_range = range(5,50,5)
classifier = GradientBoostingClassifier(n_estimators=25, learning_rate=0.1, max_features='auto', min_samples_split=20, min_samples_leaf=5)

train_scores, test_scores = validation_curve(classifier, getX(data), data['Survived'],
                                             param, param_range, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve")
plt.xlabel(param)
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
"""