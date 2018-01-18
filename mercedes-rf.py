import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


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

#mean_y_by_X0 = train.groupby(['X0'])['y'].mean()
#df_all = df_all.join(mean_y_by_X0, on='X0', how='outer', rsuffix='_mean')

df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

# Feature: Number of feats per car
#df_all['num_feats_car'] = df_all.astype(bool).sum(axis=1)
#df_all['num_feats_car'] = df_all.sum(axis=1)
#print df_all['num_feats_car'].head()
#exit()

train = df_all[:num_train]
test = df_all[num_train:]

model = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                              min_samples_leaf=25, max_depth=3)

#model = RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=20,
#                              min_samples_leaf=20, max_depth=6)

# model.fit(train, y_train)
#
# y_pred = model.predict(test)
#
# df_sub = pd.DataFrame({'ID': id_test, 'y': y_pred})
# df_sub.to_csv('mercedes_submissions/rf.csv', index=False)


cv_pred = cross_val_predict(model, train, y_train, cv=5)
print("R2 score: %.4f" % r2_score(y_train, cv_pred))
exit()

# results = cross_val_score(model, train, y_train, cv=25, scoring='r2')
# print("R2 score: %.4f (%.4f)" % (results.mean(), results.std()))

## Investigating feats-per-car-count => Hmmm ... ?
## R2 score: 0.5837 (0.1113) ; With feats-count .astype(bool)
## R2 score: 0.5833 (0.1114) ; With feats-count
## R2 score: 0.5836 (0.1113) ; Without feats-count

## Investigating magic-feat
## R2 score: -0.0094 (0.0113) ; With magick feat
## R2 score:  0.5836 (0.1113) ; Without magick feat

gsc = GridSearchCV(
    estimator=RandomForestRegressor(n_estimators=250, n_jobs=4, min_samples_split=25,
                                    min_samples_leaf=25, max_depth=6),
    param_grid={
        #'n_estimators': range(75,251,25),  # 100-150-200
        #'max_features': range(150,500,50),  # 200-250-300-350-400
        #'min_samples_leaf': range(15,30,5),  # 20-25
        #'min_samples_split': range(15,30,5),  # 15-20
        #'max_depth': range(2,6),  # 3-4
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