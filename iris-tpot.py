from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import tpot

X,y = datasets.load_iris(return_X_y=True)

#model = tpot.TPOTClassifier(generations=50, population_size=50, verbosity=3, config_dict='TPOT light')
model = tpot.TPOTClassifier(generations=50, population_size=50, verbosity=3)
model.fit(X, y)

model.export('iris-tpot-result.py')

pipe = model._toolbox.compile(expr=model._optimized_pipeline)
cv_pred = cross_val_predict(pipe, X, y, cv=5)
print("Score: %.4f" % f1_score(y, cv_pred, average='micro'))