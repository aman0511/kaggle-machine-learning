import numpy as np
import csv
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten, GaussianNoise, Dropout, Embedding, Conv1D, MaxPooling1D, InputLayer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from nltk.tokenize import wordpunct_tokenize
from keras.utils.np_utils import to_categorical


# load the dataset
#top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# X = numpy.concatenate((X_train, X_test), axis=0)
# y = numpy.concatenate((y_train, y_test), axis=0)

data = csv.reader(open('/home/dehling/Downloads/training.cleaned.csv', 'rb'), delimiter=',', quotechar='"')
data = [d for d in data]
X = [row[5] for row in data]
y = np.array([row[0] for row in data])

y = to_categorical(y)

top_words = 50000
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# summarize size
#print("The data: ")
#print(X.shape)
#print(y.shape)

# Summarize number of words
#print("Number of words: ")
#print(len(np.unique(np.hstack(X))))

# Summarize review length
print("Review length: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

max_seq_len = 20
X_train = sequence.pad_sequences(X_train, maxlen=max_seq_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_seq_len)

## create a simple model
# model = Sequential()
# model.add(Embedding(top_words, 32, input_length=max_seq_len))
# model.add(Flatten())
# model.add(Dense(250, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model = Sequential()
#model.add(InputLayer(input_shape=(None,max_seq_len,1)))
model.add(Embedding(top_words, 64, input_length=max_seq_len))
model.add(Dropout(0.25))
model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(125, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=512, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
