from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# See also maybe https://www.kaggle.com/sudhirnl7/logistic-regression-tfidf
from sklearn.pipeline import make_pipeline

#df_train = pd.read_csv('toxicity-train.csv')
df_train = pd.read_csv('toxicity-train.csv', nrows=10000)
df_test = pd.read_csv('toxicity-test.csv', nrows=5)

df_test.fillna(' ',inplace=True)

X_train = df_train['comment_text'].values
X_test = df_test['comment_text'].values



# Best {'logisticregression__class_weight': {0: 1, 1: 1.75}}


pipe = make_pipeline(
    #TfidfVectorizer(min_df=10, max_df=0.75, strip_accents='unicode', analyzer='char', ngram_range=(3,3), sublinear_tf=True),
    TfidfVectorizer(min_df=3, max_df=0.75, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                    sublinear_tf=True, stop_words='english', ngram_range=(1, 2)),
    LogisticRegression(C=7.0, class_weight={0: 1, 1: 1.75})
)

gsc = GridSearchCV(
    estimator=pipe,
    param_grid={
        #'logisticregression__class_weight': [{0:1, 1: x} for x in np.arange(0.75,2.5,0.25)],
        #'logisticregression__C': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        #'logisticregression__penalty': ['l1', 'l2'],
        #'logisticregression__dual': [True, False],
        #'tfidfvectorizer__min_df': range(2, 15),
        #'tfidfvectorizer__max_df': np.arange(0.2,1.0,0.2),
        #'tfidfvectorizer__sublinear_tf': [True, False],
        #'tfidfvectorizer__use_idf': [True, False],
        #'tfidfvectorizer__smooth_idf': [True, False],
        #'tfidfvectorizer__ngram_range': [(3,3), (4,4)],
        'tfidfvectorizer__strip_accents': [None, 'ascii', 'unicode'],
    },
    cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2),
    scoring='neg_log_loss',
    verbose=3,
    n_jobs=4
)

grid_result = gsc.fit(X_train, df_train['toxic'].values)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

for test_mean, param in zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['params']):
    print("Test : %f with: %r" % (test_mean, param))

exit()







vect = TfidfVectorizer(min_df=10, max_df=0.9, max_features=None, strip_accents='unicode',
                       analyzer='char', ngram_range=(3,3), use_idf=1, smooth_idf=1, sublinear_tf=1)
vect.fit(list(X_train) + list(X_test))

X_train = vect.transform(X_train)
X_test = vect.transform(X_test)

y_columns = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

df_out = df_test[['id']]

for y_col in y_columns:
    model = LogisticRegression(C=4.0)
    #model = LogisticRegression()
    #model = LogisticRegression(class_weight={0: 1.0, 1: 5.0})
    #model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, df_train[y_col].values)
    df_out[y_col] = model.predict_proba(X_test)[:,1]

df_out.to_csv('toxicity-submission-cng.csv', index=False)
