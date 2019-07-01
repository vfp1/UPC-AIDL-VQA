
# modification of model from https://github.com/avisingh599/visual-qa
from keras.applications.vgg16 import VGG16

# Modificaiton of file obtained from here
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
from keras import backend as K_back
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D as Convolution2D
import keras.backend as K


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

    def VGG_16_pretrained(self, input_shape=(3, 224, 224), frozen_layers=4,
                          fine_tune_dropout=0.5, fine_tune_activation='relu'):
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

        """
        Dont be an idiot and freeze the bottom (first) layers [:4]
        
        # Freeze the layers except the last 4 layers
        print("TRAINABLE LAYERS")
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        """

        # Freeze the bottom layers
        print("TRAINABLE LAYERS")
        for layer in base_model.layers[:frozen_layers]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in base_model.layers:
            print(layer, layer.trainable)

        # finetune
        fine_tuned = Sequential()
        fine_tuned.add(base_model)
        fine_tuned.add(Flatten())
        fine_tuned.add(Dense(4096, activation=fine_tune_activation))
        fine_tuned.add(Dropout(fine_tune_dropout))
        fine_tuned.add(Dense(4096, activation=fine_tune_activation))

        return fine_tuned

    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """
        # KL loss
        kl_loss = self.calculate_kl_loss
        # Reconstruction loss
        md_loss_func = self.calculate_md_loss

        # KL weight (to be used by total loss and by annealing scheduler)
        self.kl_weight = K.variable(self.hps['kl_weight_start'], name='kl_weight')
        kl_weight = self.kl_weight

        def seq2seq_loss(y_true, y_pred):
            """ Final loss calculation function to be passed to optimizer"""
            # Reconstruction loss
            md_loss = md_loss_func(y_true, y_pred)
            # Full loss
            model_loss = kl_weight*kl_loss() + md_loss
            return model_loss

        return seq2seq_loss