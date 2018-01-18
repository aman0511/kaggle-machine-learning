import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    RobustScaler(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, min_samples_split=8)),
    DecisionTreeRegressor(max_depth=3, min_samples_leaf=18, min_samples_split=19)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
