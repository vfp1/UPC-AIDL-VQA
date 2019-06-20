
# modification of model from https://github.com/avisingh599/visual-qa
from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Dense
from keras.layers.merge import Concatenate
from keras.layers import Concatenate

# Modificaiton of file obtained from here
# https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D as Convolution2D
import numpy as np

import h5py


class VGG(object):
    """

    """

    def pop(self, model):
        '''
        Removes a layer instance on top of the layer stack.
        This code is thanks to @joelthchao https://github.com/fchollet/keras/issues/2371#issuecomment-211734276
        '''
        if not model.outputs:
            raise Exception('Sequential model cannot be popped: model is empty.')
        else:
            model.layers.pop()
            if not model.layers:
                model.outputs = []
                model.inbound_nodes = []
                model.outbound_nodes = []
            else:
                model.layers[-1].outbound_nodes = []
                model.outputs = [model.layers[-1].output]
            model.built = False

        return model


    def load_model_legacy(self, model, weight_path):
        '''
        This function is used because the weights in this model
        were trained with legacy keras. New keras does not support loading these weights
        '''

        f = h5py.File(weight_path, mode='r')
        flattened_layers = model.layers

        nb_layers = f.attrs['nb_layers']

        for k in range(nb_layers):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            if not weights: continue
            if len(weights[0].shape) > 2:
                # swap conv axes
                # note np.rollaxis does not work with HDF5 Dataset array
                weights[0] = np.swapaxes(weights[0], 0, 3)
                weights[0] = np.swapaxes(weights[0], 0, 2)
                weights[0] = np.swapaxes(weights[0], 1, 2)
            flattened_layers[k].set_weights(weights)

        f.close()


    def VGG_16(self, weights_path=None):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(3, 4096, 4096)))
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
        #model.add(Dropout(0.5))
        #model.add(Dense(1000, activation='softmax'))

        if weights_path:
            # model.load_weights(weights_path)
            self.load_model_legacy(model, weights_path)

        # Remove the last two layers to get the 4096D activations
        model = self.pop(model)
        model = self.pop(model)

        return model

class VQA_model(object):
    """

    """

    def VQA_MODEL(self):
        image_feature_size = 4096
        word_feature_size = 300
        number_of_LSTM = 3
        number_of_hidden_units_LSTM = 512
        max_length_questions = 30
        number_of_dense_layers = 3
        number_of_hidden_units = 1024
        activation_function = 'tanh'
        dropout_pct = 0.5

        # Image model
        #model_image = Sequential()
        model_image = VGG().VGG_16()
        model_image.add(Reshape((image_feature_size,), input_shape=(image_feature_size,)))

        # Language Model
        model_language = Sequential()
        model_language.add(
            LSTM(number_of_hidden_units_LSTM, return_sequences=True, input_shape=(max_length_questions, word_feature_size)))
        model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=True))
        model_language.add(LSTM(number_of_hidden_units_LSTM, return_sequences=False))

        # combined model

        # model = Concatenate()([model_language, model_image])
        model = Sequential()
        # model.add(Merge([model_language, model_image], mode='concat', concat_axis=1))
        model.add(Concatenate([model_language, model_image]))

        for _ in range(number_of_dense_layers):
            model.add(Dense(number_of_hidden_units, kernel_initializer='uniform'))
            model.add(Activation(activation_function))
            model.add(Dropout(dropout_pct))

        model.add(Dense(1000))
        model.add(Activation('softmax'))

        return model