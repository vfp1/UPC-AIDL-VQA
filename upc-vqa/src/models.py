
# modification of model from https://github.com/avisingh599/visual-qa
from keras.applications.vgg16 import VGG16

# Modificaiton of file obtained from here
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
from keras import backend as K_back
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D as Convolution2D


class VGG(object):
    """
    Keras loading VGG

    """


    def VGG_16(self):
        #
        # Nota: "th" format means that the convolutional kernels will have the shape (depth, input_depth, rows, cols)
        #       "tf" format means that the convolutional kernels will have the shape (rows, cols, input_depth, depth)
        #
        # si K_back.set_image_dim_ordering('th') habría que poner cambiar el reshape y el input_shape=(1, 28, 28) en la CONV2D
        # David es el puto amo
        # If we don't put this line, we need the to put X, Y, Z instead of Z, X, Y
        K_back.set_image_dim_ordering('th')
        #

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))

        return model

    def VGG_16_pretrained(self, input_shape=(3, 224, 224)):
        #
        # Nota: "th" format means that the convolutional kernels will have the shape (depth, input_depth, rows, cols)
        #       "tf" format means that the convolutional kernels will have the shape (rows, cols, input_depth, depth)
        #
        # si K_back.set_image_dim_ordering('th') habría que poner cambiar el reshape y el input_shape=(1, 28, 28) en la CONV2D
        # David es el puto amo
        # If we don't put this line, we need the to put X, Y, Z instead of Z, X, Y
        K_back.set_image_dim_ordering('th')
        #

        base_model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')

        print('Model VGG pre loaded')
        print(base_model.summary())

        # Freeze the layers except the last 4 layers
        print("TRAINABLE LAYERS")
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in base_model.layers:
            print(layer, layer.trainable)

        # finetune
        fine_tuned = Sequential()
        fine_tuned.add(base_model)
        fine_tuned.add(Flatten())
        fine_tuned.add(Dense(4096, activation='relu'))
        fine_tuned.add(Dropout(0.5))
        fine_tuned.add(Dense(4096, activation='relu'))

        return fine_tuned