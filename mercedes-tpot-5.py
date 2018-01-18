import numpy as np

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=Lasso(alpha=0.0312727272727)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="quantile", max_depth=1, max_features=0.35, min_samples_leaf=8, min_samples_split=5, n_estimators=125, subsample=0.95)),
    AdaBoostRegressor(learning_rate=0.01)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
