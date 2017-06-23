import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline


train = pd.read_csv('mercedes_train.csv')
test = pd.read_csv('mercedes_test.csv')

y_train = train['y'].values
id_test = test['ID']

num_train = len(train)
df_all = pd.concat([train, test])
df_all.drop(['ID', 'y'], axis=1, inplace=True)

# One-hot encoding of categorical/strings
df_all = pd.get_dummies(df_all, drop_first=True)

train = df_all[:num_train]
test = df_all[num_train:]

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(train, y_train)

df_sub = pd.DataFrame({'ID': id_test, 'y': knn.predict(test)})
df_sub.to_csv('mercedes-submission.csv', index=False)
