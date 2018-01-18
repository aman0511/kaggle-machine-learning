from sklearn.linear_model import LogisticRegression
from gensim.sklearn_api.d2vmodel import D2VTransformer
from gensim.parsing.preprocessing import preprocess_string
from gensim.models import Word2Vec, Doc2Vec, KeyedVectors
import pandas as pd

df_train = pd.read_csv('toxicity-train.csv')
df_test = pd.read_csv('toxicity-test.csv')

df_test.fillna(' ',inplace=True)

X_train = df_train['comment_text'].map(preprocess_string)
X_test = df_test['comment_text'].map(preprocess_string)

d2v_model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)


vect = D2VTransformer()
vect.fit(list(X_train_w2v) + list(X_test_w2v))

X_train = vect.transform(X_train.values)
X_test = vect.transform(X_test.values)

y_columns = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

df_out = df_test[['id']]

for y_col in y_columns:
    model = LogisticRegression(C=4.0)
    model.fit(X_train, df_train[y_col].values)
    df_out[y_col] = model.predict_proba(X_test)[:,1]

df_out.to_csv('toxicity-submission-w2v.csv', index=False)
