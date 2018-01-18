import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor


train = pd.read_csv('mercedes_train.csv')
test = pd.read_csv('mercedes_test.csv')

extra = pd.read_csv('mercedes-extra.csv')
train_extra = extra.join(test, on='ID', how='inner', rsuffix='_bla')
train_extra.drop(['ID_bla'], axis=1, inplace=True)
train = pd.concat([train, train_extra])

y_train = train['y'].values
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train].values
test = df_all[num_train:].values

model = ExtraTreesRegressor(n_estimators=75, n_jobs=4, min_samples_split=25, min_samples_leaf=35,
                            max_features=150)

model.fit(train, y_train)

df_sub = pd.DataFrame({'ID': id_test, 'y': model.predict(test)})
df_sub.to_csv('mercedes_submissions/et.csv', index=False)

exit()

gsc = GridSearchCV(
    estimator=model,
    param_grid={
        #'n_estimators': range(50,126,25),
        #'max_features': range(50,401,50) + ['auto'],
        #'min_samples_leaf': range(20,50,5),
        #'min_samples_split': range(15,36,5),
    },
    scoring='r2',
    cv=5
)

grid_result = gsc.fit(train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for test_mean, train_mean, param in zip(
        grid_result.cv_results_['mean_test_score'],
        grid_result.cv_results_['mean_train_score'],
        grid_result.cv_results_['params']):
    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))