import numpy as np
import pandas
from scipy.stats import skew
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

#
# Borrowing from https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
#

train = pandas.read_csv("kaggle-houses-train.csv")
test = pandas.read_csv("kaggle-houses-test.csv")

all_data = pandas.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']))

#log transform the price:
train["SalePrice"] = np.log1p(train["SalePrice"])

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

all_data = pandas.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

#log transform skewed numeric features:
skewness = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
left_skewed_feats = skewness[skewness > 0.5].index
right_skewed_feats = skewness[skewness < -0.5].index
all_data[left_skewed_feats] = np.log1p(all_data[left_skewed_feats])
#all_data[right_skewed_feats] = np.exp(all_data[right_skewed_feats])

scaler = RobustScaler()
all_data[numeric_feats] = scaler.fit_transform(all_data[numeric_feats])

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train['SalePrice']

linear_model = ElasticNet(alpha=0.001)
linear_model.fit(X_train, y)

svr_model = SVR(kernel='rbf', C=2, epsilon=0.05)
svr_model.fit(X_train, y)

test['SalePrice'] = np.expm1((linear_model.predict(X_test) +
                              svr_model.predict(X_test)) / 2.0)

test.to_csv('kaggle-houses-submission.csv', index=False, columns=['Id', 'SalePrice'])
