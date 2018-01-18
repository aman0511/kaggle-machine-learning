from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# See also maybe https://www.kaggle.com/sudhirnl7/logistic-regression-tfidf

df_train = pd.read_csv('toxicity-train.csv')
df_test = pd.read_csv('toxicity-test.csv')

df_test.fillna(' ',inplace=True)

X_train = df_train['comment_text'].values
X_test = df_test['comment_text'].values

#vect = TfidfVectorizer(min_df=250, max_df=0.2, stop_words='english')
vect = TfidfVectorizer(min_df=3, max_df=0.9, max_features=None, strip_accents='unicode', \
                       analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), use_idf=1, \
                       smooth_idf=1, sublinear_tf=1, stop_words='english')
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

df_out.to_csv('toxicity-submission.csv', index=False)
