from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# See also maybe https://www.kaggle.com/sudhirnl7/logistic-regression-tfidf

df_train = pd.read_csv('toxicity-train.csv')
df_test = pd.read_csv('toxicity-test.csv')

df_test.fillna(' ',inplace=True)

X_train = df_train['comment_text'].values
X_test = df_test['comment_text'].values

vect = TfidfVectorizer(min_df=10, max_df=0.75, strip_accents='unicode', analyzer='char', ngram_range=(3,3), sublinear_tf=True)
vect.fit(list(X_train) + list(X_test))

X_train = vect.transform(X_train)
X_test = vect.transform(X_test)

y_columns = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

df_out = df_test[['id']]

for y_col in y_columns:
    model = LogisticRegression(C=6.0, class_weight={0: 1.0, 1: 1.75})
    model.fit(X_train, df_train[y_col].values)
    df_out[y_col] = model.predict_proba(X_test)[:,1]

df_out.to_csv('toxicity-submission-cng.csv', index=False)
