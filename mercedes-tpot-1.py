import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=19, min_samples_split=4)),
    VarianceThreshold(),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=12, min_samples_split=11)),
    DecisionTreeRegressor(max_depth=3, min_samples_leaf=12, min_samples_split=20)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
