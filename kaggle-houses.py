import pandas
import numpy as np
from scipy.stats import skew
from math import fabs

import matplotlib
import matplotlib.pyplot as plt



from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler

from xgboost import XGBRegressor


#
# Borrowing from https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
#

train = pandas.read_csv("kaggle-houses-train.csv")
test = pandas.read_csv("kaggle-houses-test.csv")


all_data = pandas.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']))


#all_data.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
#               'Heating', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt',
#               'GarageArea', 'GarageCond', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
#               'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'], axis=1, inplace=True)


#log transform the price:
train["SalePrice"] = np.log1p(train["SalePrice"])



# Impute LotArea values
x = all_data.loc[np.logical_not(all_data["LotFrontage"].isnull()), "LotArea"]
y = all_data.loc[np.logical_not(all_data["LotFrontage"].isnull()), "LotFrontage"]
#plt.scatter(x, y)
#plt.show()
t = (x <= 25000) & (y <= 150)
p = np.polyfit(x[t], y[t], 1)
all_data.loc[all_data['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, all_data.loc[all_data['LotFrontage'].isnull(), 'LotArea'])

# # Impute other missing/bad values
# all_data.loc[all_data.Alley.isnull(), 'Alley'] = 'NoAlley'
# all_data.loc[all_data.MasVnrType.isnull(), 'MasVnrType'] = 'None' # no good
# all_data.loc[all_data.MasVnrType == 'None', 'MasVnrArea'] = 0
# all_data.loc[all_data.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'
# all_data.loc[all_data.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'
# all_data.loc[all_data.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'
# all_data.loc[all_data.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'
# all_data.loc[all_data.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'
# all_data.loc[all_data.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0
# all_data.loc[all_data.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0
# all_data.loc[all_data.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = all_data.BsmtFinSF1.median()
# all_data.loc[all_data.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0
# all_data.loc[all_data.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = all_data.BsmtUnfSF.median()
# all_data.loc[all_data.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0
# all_data.loc[all_data.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'
# all_data.loc[all_data.GarageType.isnull(), 'GarageType'] = 'NoGarage'
# all_data.loc[all_data.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'
# all_data.loc[all_data.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'
# all_data.loc[all_data.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'
# all_data.loc[all_data.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0
# all_data.loc[all_data.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0
# all_data.loc[all_data.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
# all_data.loc[all_data.MSZoning.isnull(), 'MSZoning'] = 'RL'
# all_data.loc[all_data.Utilities.isnull(), 'Utilities'] = 'AllPub'
# all_data.loc[all_data.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'
# all_data.loc[all_data.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'
# all_data.loc[all_data.Functional.isnull(), 'Functional'] = 'Typ'
# all_data.loc[all_data.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'
# all_data.loc[all_data.SaleCondition.isnull(), 'SaleType'] = 'WD'
# all_data.loc[all_data['PoolQC'].isnull(), 'PoolQC'] = 'NoPool'
# all_data.loc[all_data['Fence'].isnull(), 'Fence'] = 'NoFence'
# all_data.loc[all_data['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
# all_data.loc[all_data['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
# # only one is null and it has type Detchd
# all_data.loc[all_data['GarageArea'].isnull(), 'GarageArea'] = all_data.loc[all_data['GarageType']=='Detchd', 'GarageArea'].mean()
# all_data.loc[all_data['GarageCars'].isnull(), 'GarageCars'] = all_data.loc[all_data['GarageType']=='Detchd', 'GarageCars'].median()


#Checking for missing data
#NAs = pandas.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
#NAs[NAs.sum(axis=1) > 0]
#print NAs
#exit()

# MSSubClass as str
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

# MSZoning NA in pred. filling with most popular values
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# Alley  NA in all. NA means no access
all_data['Alley'] = all_data['Alley'].fillna('NoAlley')

# Converting OverallCond to str
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

# MasVnrType NA in all. filling with most popular values
all_data['MasVnrType'] = all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('NoBSMT')

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)

# Electrical NA in pred. filling with most popular values
#all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# KitchenAbvGr to categorical
all_data['KitchenAbvGr'] = all_data['KitchenAbvGr'].astype(str)

# KitchenQual NA in pred. filling with most popular values
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

# FireplaceQu  NA in all. NA means No Fireplace
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('NoFP')

# GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage
for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    all_data[col] = all_data[col].fillna('NoGRG')

# GarageCars  NA in pred. I suppose NA means 0
all_data['GarageCars'] = all_data['GarageCars'].fillna(0.0)

# SaleType NA in pred. filling with most popular values
#all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# Year and Month to categorical
#all_data['YrSold'] = all_data['YrSold'].astype(str)
#all_data['MoSold'] = all_data['MoSold'].astype(str)

# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)



#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

all_data = pandas.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

skewness = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
left_skewed_feats = skewness[skewness > 0.5].index
right_skewed_feats = skewness[skewness < -0.5].index
all_data[left_skewed_feats] = np.log1p(all_data[left_skewed_feats])
#all_data[right_skewed_feats] = np.exp(all_data[right_skewed_feats])


#def plot_histograms(col):
#    pandas.DataFrame({'value': col,
#                      'log(value + 1)': np.log1p(col),
#                      'exp(value + 1)': np.log1p(col)}).hist()
#    plt.show()
#plot_histograms(train['SalePrice'])
#plot_histograms(train['LotArea'])
#exit()


scaler = RobustScaler()
all_data[numeric_feats] = scaler.fit_transform(all_data[numeric_feats])

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train['SalePrice']

#scaler = RobustScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)


# # Determine best alpha value
# alphas = [0.0005, 0.00075, 0.001, 0.0011, 0.00125, 0.0015]
# cv_ridge = [
#     np.sqrt(-cross_val_score(ElasticNet(alpha), X_train, y, scoring="neg_mean_squared_error", cv=5)).mean()
#     for alpha in alphas
# ]
# cv_ridge = pandas.Series(cv_ridge, index=alphas)
# cv_ridge.plot(title = "Alphas")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()
# exit()


gsc = GridSearchCV(
     estimator=SVR(kernel='rbf', epsilon=0.05),
     param_grid={
         'C': range(1,4),
         'epsilon': (0.02, 0.04, 0.06, 0.08),
     },
     # estimator=ElasticNet(),
     # param_grid={
     #     'alpha': (0.01, 0.02, 0.03, 0.04, 0.05),
     #     'l1_ratio': (0.2, 0.4, 0.6, 0.8),
     # },
     cv=5
)
grid_result = gsc.fit(X_train, y)

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for test_mean, test_stdev, train_mean, train_stdev, param in zip(
#          grid_result.cv_results_['mean_test_score'],
#          grid_result.cv_results_['std_test_score'],
#          grid_result.cv_results_['mean_train_score'],
#          grid_result.cv_results_['std_train_score'],
#          grid_result.cv_results_['params']):
#      print("Train: %f (%f) // Test : %f (%f) with: %r" % (train_mean, train_stdev, test_mean, test_stdev, param))

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter([row['C'] for row in grid_result.cv_results_['params']],
           [row['epsilon'] for row in grid_result.cv_results_['params']],
           grid_result.cv_results_['mean_test_score'],
           c='b', marker='^')

ax.set_xlabel('C')
ax.set_ylabel('Epsilon')
ax.set_zlabel('Score')

plt.show()

exit()


models = [
    #('Ridge(alpha=8)', Ridge(alpha=8)),
    #('Ridge(alpha=10)', Ridge(alpha=10)),
    #('Ridge(alpha=12)', Ridge(alpha=12)),

    #('RidgeCV(alphas=..., cv=5)', RidgeCV(alphas=range(0,20,2), cv=5)),

    #('LassoCV(cv=10)', LassoCV(alphas=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], cv=10)),

    ('ElasticNet(alpha=0.001)', ElasticNet(alpha=0.001)),

    ('SVR(kernel=\'rbf\', C=3, epsilon=0.05)', SVR(kernel='rbf', C=3, epsilon=0.05)),

    #('RandomForestRegressor(n_estimators=400)', RandomForestRegressor(n_estimators=400, max_features=75)),

    #('GradientBoostingRegressor()', GradientBoostingRegressor(n_estimators=250, max_features='auto', min_samples_split=75, min_samples_leaf=5, max_depth=3)),

    #('MLPRegressor((50, 50,))', MLPRegressor((50, 50,))),
    #('MLPRegressor(50)', MLPRegressor(hidden_layer_sizes=(50,))),
    #('MLPRegressor(100)', MLPRegressor(hidden_layer_sizes=(100,))),
    #('MLPRegressor(250,250)', MLPRegressor(hidden_layer_sizes=(250,250,))),
    #('MLPRegressor(500,500)', MLPRegressor(hidden_layer_sizes=(500,500,), max_iter=1000)),

    #('XGBoost()', XGBRegressor(n_estimators=350, learning_rate=0.1, subsample=0.8, max_depth=3, reg_alpha=0.1, reg_lambda=0.75, gamma=0.005)),
]

for name, model in models:
    model.fit(X_train, y)
    print name + ' : '
    # Mean squared error, less is better
    print np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=10)).mean()
    #print model.score(X_train, y)
    print
exit()


linear_model = ElasticNet(alpha=0.001)
linear_model.fit(X_train, y)

svr_model = SVR(kernel='rbf', C=3, epsilon=0.05)
svr_model.fit(X_train, y)

#rf_model = RandomForestRegressor(n_estimators=400, max_features=75)
#rf_model.fit(X_train, y)

#gb_model = GradientBoostingRegressor(n_estimators=250, max_features='auto', min_samples_split=75, min_samples_leaf=5, max_depth=3)
#gb_model.fit(X_train, y)

#xgb_model = XGBRegressor(n_estimators=350, learning_rate=0.1, subsample=0.8, max_depth=3, reg_alpha=0.1, reg_lambda=0.75, gamma=0.005)
#xgb_model.fit(X_train, y)

#test['SalePrice'] = np.expm1((linear_model.predict(X_test) +
#                              svr_model.predict(X_test) +
#                              rf_model.predict(X_test) +
#                              xgb_model.predict(X_test) +
#                              gb_model.predict(X_test)) / 5.0)

test['SalePrice'] = np.expm1((linear_model.predict(X_test) +
                              svr_model.predict(X_test)) / 2.0)

test.to_csv('kaggle-houses-submission.csv', index=False, columns=['Id', 'SalePrice'])


# Plot residuals
#preds = pandas.DataFrame({"preds": model.predict(X_train), "true":y})
#preds["residuals"] = preds["true"] - preds["preds"]
#preds.plot(x="preds", y="residuals", kind="scatter")
#plt.show()