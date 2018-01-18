from keras.optimizers import adam
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from gensim.parsing.preprocessing import preprocess_string
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df_train = pd.read_csv('toxicity-train.csv')
#df_train = pd.read_csv('toxicity-train.csv', nrows=5000)

df_test = pd.read_csv('toxicity-test.csv')
#df_test = pd.read_csv('toxicity-test.csv', nrows=3000)
df_test.fillna('', inplace=True)

print("Train shape = (%d, %d)" % df_train.shape)
print("Test shape = (%d, %d)" % df_test.shape)

X_train = df_train['comment_text']
X_test = df_test['comment_text']

y_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = df_train[y_columns].values

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15)

def count_caps(comment):
    return sum(1 for c in comment if c.isupper())

def caps_vs_total_length(comment):
    return float(count_caps(comment))/float(len(comment)+1)

chars = TfidfVectorizer(min_df=10, max_df=0.75, strip_accents='ascii', analyzer='char', ngram_range=(3, 3), sublinear_tf=True)
chars.fit(list(X_train.values) + list(X_test.values))

#words = TfidfVectorizer(min_df=50, max_df=0.75, strip_accents='unicode', analyzer='word', tokenizer=preprocess_string, sublinear_tf=True, stop_words='english', ngram_range=(1, 2))
words = TfidfVectorizer(min_df=50, max_df=0.75, strip_accents='unicode', analyzer='word', tokenizer=preprocess_string, sublinear_tf=True, stop_words='english')
words.fit(list(X_train.values) + list(X_test.values))

def calc_feature_sparse(data, feature_function):
    return csr_matrix(np.reshape(data.map(feature_function).values, (data.shape[0], 1)))

X_train = hstack((
    chars.transform(X_train.values),
    words.transform(X_train.values),
    calc_feature_sparse(X_train, caps_vs_total_length),
), format='csr')

X_test = hstack((
    chars.transform(X_test.values),
    words.transform(X_test.values),
    calc_feature_sparse(X_test, caps_vs_total_length),
), format='csr')

print("Train shape = (%d, %d)" % X_train.shape)
print("Test shape = (%d, %d)" % X_test.shape)


def batch_generator(X, y, batch_size):
    n_splits = X.shape[0] // (batch_size - 1)
    X = np.array_split(X, n_splits)
    y = np.array_split(y, n_splits)

    while True:
        for i in range(X.shape[0]):
            X_batch = []
            y_batch = []
            for ii in range(len(X.shape[0])):
                X_batch.append(X[i][ii].toarray().astype(np.int8)) # conversion sparse matrix -> np.array
                y_batch.append(y[i][ii])
            yield (np.array(X_batch), np.array(y_batch))


def batch_generator2(X, y, batch_size):
    number_of_batches = X.shape[0]/batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X =  X[shuffle_index, :]
    y =  y[shuffle_index]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[index_batch,:].todense()
        y_batch = y[index_batch]
        counter += 1
        yield(np.array(X_batch),y_batch)
        if (counter < number_of_batches):
            np.random.shuffle(shuffle_index)
            counter=0

def batch_generator3(X, batch_size):
    #number_of_batches = X.shape[0]/batch_size
    indices = np.arange(np.shape(X)[0])
    batch=0
    while 1:
        index_batch = indices[batch_size*batch : batch_size*(batch+1)]
        X_batch = X[index_batch,:].todense()
        batch += 1

        yield np.array(X_batch)


from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer


model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1],)))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(6, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=adam(decay=0.001), metrics=['accuracy'])

print(model.summary())

BATCH_SIZE = 1024

# model.fit(xtr, ytr, validation_data=(xte, yte), epochs=10, batch_size=128, verbose=2)

# model.fit_generator(batch_generator2(X_train, y_train, BATCH_SIZE), steps_per_epoch=X_train.shape[0]/BATCH_SIZE,
#                     epochs=4, verbose=1, validation_data=(X_test.todense(), y_test))
#
# scores = model.evaluate(X_test.todense(), y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# exit()


# model.fit(X_train, y_train, epochs=2, batch_size=128, verbose=2)

model.fit_generator(batch_generator2(X_train, y_train, BATCH_SIZE), steps_per_epoch=X_train.shape[0]/BATCH_SIZE,
                    epochs=4, verbose=1)

y_test = model.predict_generator(batch_generator3(X_test, BATCH_SIZE), steps=X_test.shape[0]/BATCH_SIZE)

# y_test = model.predict(X_test)
sample_submission = pd.read_csv("toxicity-sample_submission.csv")
sample_submission[y_columns] = y_test
sample_submission.to_csv("toxicity-submission-keras.csv", index=False)
