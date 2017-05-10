import cPickle

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dropout, Dense, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

import numpy as np

K.set_image_dim_ordering('th') #input shape: (channels, height, width)

with open('cifar-10-batches-py/data_batch_1', 'r') as file:
    batch1 = cPickle.load(file)

with open('cifar-10-batches-py/data_batch_2', 'r') as file:
    batch2 = cPickle.load(file)

with open('cifar-10-batches-py/data_batch_3', 'r') as file:
    batch3 = cPickle.load(file)

with open('cifar-10-batches-py/data_batch_4', 'r') as file:
    batch4 = cPickle.load(file)

with open('cifar-10-batches-py/data_batch_5', 'r') as file:
    batch5 = cPickle.load(file)

def combine(first, *mores):
    data = first
    for more in mores:
        data = np.append(data, more, axis=0)
    return data

x_train = combine(batch1['data'], batch2['data'], batch3['data'], batch4['data'], batch5['data'])
x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
x_train = x_train.astype('float32')/255

y_train = combine(batch1['labels'], batch2['labels'], batch3['labels'], batch4['labels'], batch5['labels'])
y_train = to_categorical(y_train)


def build_model(dense_size=750, dropout1=0.2, dropout2=0.35, conv_filters=16, conv_sz=3, optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(filters=conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu', input_shape=x_train.shape[1:]))
    model.add(Conv2D(filters=2 * conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=4 * conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu'))
    model.add(Conv2D(filters=8 * conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=16 * conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu'))
    model.add(Conv2D(filters=16 * conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu'))
    #model.add(Dropout(dropout1))
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dropout(dropout2))
    model.add(Dense(10, activation='softmax', activity_regularizer='l1_l2'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

param_grid = dict(
    #dense_size=(250,500,750),
    #dropout1=(0.0, 0.2, 0.4),
    #dropout2=(0.20, 0.35, 0.50),
    #conv_filters=(16,32),
    #conv_sz=(3),
    #optimizer=('adam', 'sgd', 'rmsprop'),
)

gsc = GridSearchCV(estimator=KerasClassifier(build_fn=build_model,
                                             batch_size=128,
                                             epochs=1,
                                             verbose=2),
                   param_grid=param_grid)

grid_result = gsc.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

exit()

model.fit(x_train,
          y_train,
          epochs=25,
          verbose=2,
          validation_split=.2)

#
# Ideas: https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
#
# http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
#
