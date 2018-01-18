import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# # NOTE: Make sure that the class is labeled 'class' in the data file
# tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
# features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
# training_features, testing_features, training_target, testing_target = \
#     train_test_split(features, tpot_data['class'], random_state=42)

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

exported_pipeline = make_pipeline(
    StandardScaler(),
    SelectFromModel(estimator=Lasso(alpha=0.576181818182)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=14, min_samples_split=5)),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.001, loss="square", n_estimators=100)),
    Lasso(alpha=0.303727272727)
)

exported_pipeline.fit(train, y_train)
results = exported_pipeline.predict(test)

df_sub = pd.DataFrame({'ID': id_test, 'y': results})
df_sub.to_csv('mercedes_submissions/tpot3.csv', index=False)