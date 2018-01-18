import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from pysmac.optimize import fmin
import lightgbm as lgbm

train = pd.read_csv("kaggle-houses-train.csv")
test = pd.read_csv("kaggle-houses-test.csv")

num_train = len(train)
y_train = train['SalePrice']
id_test = test['Id']

all_data = pd.concat((train, test))
all_data.drop(['Id', 'SalePrice'], inplace=True)

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X_train = all_data[:num_train]
X_test = all_data[num_train:]

def mk_model(x, x_int):
    return lgbm.LGBMRegressor(
        objective='regression',
        n_estimators=100,
        learning_rate=x[0],
        num_leaves=x_int[0],
        min_data_in_leaf=x_int[1],
        colsample_bytree=x[1],
        subsample=0.95,
        subsample_freq=1
    )

def objective(x, x_int):
    model = mk_model(x, x_int)
    return -1.0 * cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()

parameters, score = fmin(objective=objective,
                         x0=[0.2, 0.9],
                         xmin=[0.001, 0.001],
                         xmax=[0.5, 1.0],
                         x0_int=[24, 24],
                         xmin_int=[10, 10],
                         xmax_int=[50, 50],
                         max_evaluations=50)

print('Lowest function value found: %f' % score)
print('Parameter setting %s' % parameters)

model = mk_model(parameters['x'], parameters['x_int'])

model.fit(X_train, y_train)

df_sub = pd.DataFrame({'Id': id_test, 'SalePrice': model.predict(X_test)})
df_sub.to_csv('kaggle-houses-submission.csv', index=False)
