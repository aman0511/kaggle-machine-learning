import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten, InputLayer, Conv2D, MaxPooling2D, GaussianNoise
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

K.set_image_dim_ordering('th') #input shape: (channels, height, width)

train_df = pd.read_csv("kaggle-mnist-train.csv")
valid_df = pd.read_csv("kaggle-mnist-test.csv")

#train_df = train_df.ix[0:2500]

x_train = train_df.drop(['label'], axis=1).values.astype('float32')
x_valid = valid_df.values.astype('float32')

img_width, img_height = 28, 28

n_train = x_train.shape[0]
n_valid = x_valid.shape[0]

x_train = x_train.reshape(n_train,1,img_width,img_height)
x_valid = x_valid.reshape(n_valid,1,img_width,img_height)

x_train = x_train/255 #normalize from [0,255] to [0,1]
x_valid = x_valid/255

y_train = to_categorical(train_df['label'].values)

#model = Sequential()
#model.add(Conv2D(filters=40, kernel_size=(5,5), activation='relu', batch_input_shape=(None, 1, img_width, img_height)))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Conv2D(filters=40, kernel_size=(5,5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Flatten())
#model.add(Dense(1000, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1000, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation='softmax', activity_regularizer='l1_l2'))
#model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])


def build_model(noise=0.5):
    model = Sequential()

    model.add(InputLayer(batch_input_shape=(None, 1, img_width, img_height)))
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(GaussianNoise(noise))
    #model.add(Dropout(dropout))
    model.add(Dense(750, activation='relu'))
    model.add(Dense(10, activation='softmax', activity_regularizer='l1_l2'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

param_grid = dict(
    #dense_size=(400,700,1000),
    #dense_layers=(1,2),
    #dropout=(0.35,0.50),
    #conv_layers=(1,2),
    #conv_filters=(60),
    #conv_sz=(5),
    noise=[x/20.0 for x in range(1, 11)]
)

# gsc = GridSearchCV(estimator=KerasClassifier(build_fn=build_model,
#                                              batch_size=128,
#                                              epochs=1,
#                                              verbose=2),
#                    param_grid=param_grid)
#
# grid_result = gsc.fit(x_train[:2500], y_train[:2500])
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# # Best: 0.943211 using {'conv_filters': 60, 'dropout': 0.35, 'dense_size': 600}
# exit()

model = build_model()

epochs = 25
plot_start = 5

history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=epochs,
                    verbose=2,
                    validation_split=.2)

dataz = pd.DataFrame({ 'loss': history.history['loss'][plot_start:],
                       'val_loss': history.history['val_loss'][plot_start:],
                       'epochs': range(plot_start+1,epochs+1) })
dataz.plot(x='epochs')
plt.show()
exit()

yPred = model.predict_classes(x_valid,batch_size=32,verbose=1)

np.savetxt('kaggle-mnist-predictions.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
