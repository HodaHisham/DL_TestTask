import keras
import numpy as np
import matplotlib as mt

from keras import backend as K
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.utils import plot_model

mt.use('TkAgg')
K.set_image_dim_ordering('tf')

# loading input data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_train, dep, rows, cols =  x_train.shape
num_test, _, _, _ =  x_test.shape

num_classes = len(np.unique(y_train))

# normalization
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# convert to class labels
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


def build_model(size):
    # build the model
    model = Sequential()
    model.add(Conv2D(6, (size, size), padding='same', input_shape=x_train.shape[1:]))

    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Activation("sigmoid"))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(16, (size, size), padding='same'))

    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.5))

    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(120, (size, size), padding='same'))

    model.add(Flatten())

    model.add(Dense(84))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.0003)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=1)

    return model, history

def extract_1stL(model):
    first_conv = K.function(model.inputs, [model.layers[1].output])
    first_conv_out = first_conv([x_test])
    for idx in range(10):
        fig, axes = plt.subplots(ncols=6, figsize=(6, 6))
        for f in range(6):
            axes[f].imshow(first_conv_out[0][idx][f])
    plt.show()
    plt.close(fig)

def plot_history(train_value, test_value, s):
    f, axes = plt.subplots()
    axes.plot(train_value, 'o-', label="Training score")
    axes.plot(test_value, 'x-', label="Validation score")
    axes.legend(loc = 0) 
    axes.set_title('Training/Validation ' + s + ' per Epoch')
    axes.set_xlabel('Epoch')
    axes.set_ylabel(s)  

def init(sz):
    model, history = build_model(sz)
    test = model.evaluate(x_test, y_test, verbose=1)
    print "\nTest loss:", str(test[0])
    print "Test accuracy:", str(test[1])
    plot_history(history.history['loss'], history.history['val_loss'], 'Loss')
    plot_history(history.history['acc'], history.history['val_acc'], 'Accuracy')
    extract_1stL(model)

init(11)
init(5)