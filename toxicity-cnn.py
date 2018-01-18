from keras.layers import Dense, Dropout, Flatten, LSTM, Conv1D, MaxPooling1D, GlobalMaxPool1D, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from gensim.parsing.preprocessing import preprocess_string, strip_multiple_whitespaces, strip_numeric, strip_short,\
    remove_stopwords, strip_non_alphanum
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import codecs

df_train = pd.read_csv('toxicity-train.csv')
df_test = pd.read_csv('toxicity-test.csv')

df_test.fillna(' ',inplace=True)


MY_FILTERS = [
    lambda x: x.lower(), strip_non_alphanum,
    strip_multiple_whitespaces, strip_numeric,
    remove_stopwords, strip_short
]


X_train = df_train['comment_text'].map(lambda s: preprocess_string(s, MY_FILTERS)).map(' '.join).values
X_test = df_test['comment_text'].map(lambda s: preprocess_string(s, MY_FILTERS)).map(' '.join).values

y_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = df_train[y_columns].values

## From https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

# TODO https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# TODO https://www.kaggle.com/vsmolyakov/keras-cnn-with-fasttext-embeddings/output



MAX_WORDS = 20000  # None
MAX_SEQ_LEN = 48
EMBED_DIM = 300



tok = Tokenizer(num_words=MAX_WORDS)

tok.fit_on_texts(list(X_train) + list(X_test))

X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)


# Print some basic info
print("Average training sentence length : %.1f" % (float(sum(len(sentence) for sentence in X_train)) / float(len(X_train))))
print("Average test sentence length : %.1f" % (float(sum(len(sentence) for sentence in X_test)) / float(len(X_test))))
print("Number of word counts : %d" % len(tok.word_counts.keys()))
print("Number of word indices : %d" % len(tok.word_index.keys()))
print("Number of words appearing more than once : %d" % sum(1 for k, v in tok.word_counts.items() if v > 1))
print("Number of words appearing more than twice : %d" % sum(1 for k, v in tok.word_counts.items() if v > 2))
print("Number of words appearing more than three times : %d" % sum(1 for k, v in tok.word_counts.items() if v > 3))
print("Number of words appearing more than four times : %d" % sum(1 for k, v in tok.word_counts.items() if v > 4))
print("Number of words appearing more than five times : %d" % sum(1 for k, v in tok.word_counts.items() if v > 5))

X_train = pad_sequences(X_train, maxlen=MAX_SEQ_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_SEQ_LEN)


#load embeddings
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('wiki-news-300d-1M.vec', encoding='utf-8')
for line in f:
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))


#embedding matrix
print('preparing embedding matrix...')
words_not_found = []
nb_words = min(MAX_WORDS, len(tok.word_index))
embedding_matrix = np.zeros((nb_words, EMBED_DIM))
for word, i in tok.word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of good word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) != 0))
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
print('number words not found: %d' % len(words_not_found))
print("sample words not found: ", np.random.choice(words_not_found, 10))
print("Embeddings shape : (%d, %d)" % embedding_matrix.shape)
#exit()


def mlp_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, 32, input_length=MAX_SEQ_LEN))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def lstm_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, 32, input_length=MAX_SEQ_LEN))
    model.add(Dropout(0.25))
    model.add(LSTM(64))
    model.add(Dropout(0.25))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, 32, input_length=MAX_SEQ_LEN))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cnn_fasttext_model():
    FILTERS=48
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBED_DIM, weights=[embedding_matrix], input_length=MAX_SEQ_LEN, trainable=False))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=FILTERS, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=FILTERS, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=FILTERS, kernel_size=5, padding='same', activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.25))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model


# print("training CNN ...")
# model = Sequential()
# model.add(Embedding(nb_words, embed_dim,
#                     weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
# model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
# model.add(MaxPooling1D(2))
# model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
# model.add(GlobalMaxPooling1D())
# model.add(Dropout(0.5))
# model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)
#
# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# model.summary()


def embed_cnn_lstm_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, 32, input_length=MAX_SEQ_LEN))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(LSTM(64))
    model.add(Dropout(0.25))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model






model = cnn_fasttext_model()
print(model.summary())


BATCH_SIZE=1024
EPOCHS=6



# Test / evaluation
# xtr, xte, ytr, yte = train_test_split(X_train, y_train, test_size=0.15)
# model.fit(xtr, ytr, validation_data=(xte, yte), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
# scores = model.evaluate(xte, yte, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# exit()



# For real
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
y_test = model.predict(X_test)
sample_submission = pd.read_csv("toxicity-sample_submission.csv")
sample_submission[y_columns] = y_test
sample_submission.to_csv("toxicity-submission-cnn.csv", index=False)
