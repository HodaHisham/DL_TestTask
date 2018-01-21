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

# mt.use('TkAgg') # Segmentation issue with MacOS
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
    
    # Trial Alternatives:
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(16, (size, size), padding='same'))

    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Activation("sigmoid"))

    # Trial Alternatives (continued):
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(120, (size, size), padding='same'))

    model.add(Flatten())

    model.add(Dense(84))
    model.add(Activation("sigmoid"))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.0003)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_test, y_test), verbose=1)

    return model, history

# method to extract the 6 filters that are outputs of the first layer
def extract_1stL(model):
    w = model.layers[0].get_weights()
    # for idx in range(10):
    fig, axes = plt.subplots(ncols=6, figsize=(6, 6))
    x, y, z, num_filters = w[0].shape

    for f in range(num_filters):
        grid = [[[w[0][i][j][k][f] for k in range(z)] for j in range(y)] for i in range(x)]
        axes[f].imshow(grid)
    plt.show()
    plt.close(fig)

# method to plot training/validation curves using pyplot
def plot_history(train_value, test_value, s):
    f, axes = plt.subplots()
    axes.plot(train_value, 'o-', label="Training score")
    axes.plot(test_value, 'x-', label="Validation score")
    axes.legend(loc = 0) 
    axes.set_title('Training/Validation ' + s + ' per Epoch')
    axes.set_xlabel('Epoch')
    axes.set_ylabel(s)  

# initialize the model, plot curves and extract filters
def init(sz):
    model, history = build_model(sz)
    # test = model.evaluate(x_test, y_test, verbose=1)
    # print "\nTest loss:", str(test[0])
    # print "Test accuracy:", str(test[1])
    # plot_history(history.history['loss'], history.history['val_loss'], 'Loss')
    # plot_history(history.history['acc'], history.history['val_acc'], 'Accuracy')
    extract_1stL(model)

# Net 1
init(5)
# Net 2
init(11)
