#!/usr/bin/env python

__author__ = "Victor Pajuelo Madrigal"
__copyright__ = "Copyright 2019, UPC Group"
__credits__ = ["Victor Pajuelo Madrigal", "Jiasen Lu", "@abhshkdz"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Victor Pajuelo Madrigal"
__email__ = "-"
__status__ = "Development"

import sys, warnings
warnings.filterwarnings("ignore")
from random import shuffle, sample
import pickle as pk
import gc

import os

import numpy as np
import pandas as pd
import scipy.io
from keras.models import Sequential, Model
from keras.layers import concatenate
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from progressbar import Bar, ETA, Percentage, ProgressBar
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import spacy
#from spacy.en import English
from features import *
from utils import *

from models import *

import graphviz
import pydot_ng as pydot
pydot.find_graphviz()

from keras.utils import plot_model
from keras.callbacks import TensorBoard

class VQA_train(object):
    """

    """

    def train(self, data_folder):
        """

        :param data_folder:
        :return:
        """

        # Point to preprocessed data
        training_questions = open(os.path.join(data_folder, "preprocessed/ques_val.txt"),"rb").read().decode('utf8').splitlines()
        training_questions_len = open(os.path.join(data_folder,"preprocessed/ques_val_len.txt"),"rb").read().decode('utf8').splitlines()
        answers_train = open(os.path.join(data_folder,"preprocessed/answer_val.txt"),"rb").read().decode('utf8').splitlines()
        images_train = open(os.path.join(data_folder,"preprocessed/val_images_coco_id.txt"),"rb").read().decode('utf8').splitlines()
        img_ids = open(os.path.join(data_folder,'preprocessed/coco_vgg_IDMap.txt')).read().splitlines()
        #img_ids = open(os.path.join(data_folder,'preprocessed/val_images_coco_id.txt')).read().splitlines()
        vgg_path = os.path.join(data_folder, "vgg_weights/vgg_feats.mat")

        # Load english dictionary
        #nlp = spacy.load("en")
        try:
            nlp = spacy.load("en_core_web_md")
        except:
            nlp = spacy.load("en_core_web_sm")
        print ("Loaded WordVec")

        # Load VGG weights

        vgg_features = scipy.io.loadmat(vgg_path)
        img_features = vgg_features['feats']
        id_map = dict()
        print ("Loaded VGG Weights")

        # Number of most frequently occurring answers in COCOVQA (Covering >80% of the total data)
        upper_lim = 1000

        # Loading most frequent occurring answers
        training_questions, answers_train, images_train = freq_answers(training_questions,
                                                               answers_train, images_train, upper_lim)

        training_questions_len, training_questions, answers_train, images_train = (list(t) for t in zip(*sorted(zip(training_questions_len,
                                                                                                          training_questions, answers_train,
                                                                                                          images_train))))
        # Sanity check
        print("A sanity check: Lenght training questions:", len(training_questions))
        print("Lenght answers", len(answers_train))
        print("Lenght number images", len(images_train))

        # Encoding answers
        lbl = LabelEncoder()
        lbl.fit(answers_train)
        nb_classes = len(list(lbl.classes_))
        print("Number of classes:", nb_classes)
        pk.dump(lbl, open(os.path.join(data_folder, "output/label_encoder_lstm.sav"),'wb'))

        # Setting Hyperparameters

        batch_size = 256
        img_dim = 4096
        word2vec_dim = 96
        #max_len = 30 # Required only when using Fixed-Length Padding

        num_hidden_nodes_mlp = 1024
        num_hidden_nodes_lstm = 512
        num_layers_mlp = 3
        num_layers_lstm = 3
        dropout = 0.5
        activation_mlp = 'tanh'

        num_epochs = 90
        log_interval = 15

        for ids in img_ids:
            id_split = ids.split()
            id_map[id_split[0]] = int(id_split[1])

        #-------------------------------------------------------------------------------------------------
        # Image model, a very simple MLP
        image_model = Sequential()
        image_model.add(Dense(num_hidden_nodes_mlp, input_dim = img_dim, kernel_initializer='uniform'))
        image_model.add(Dropout(dropout))

        for i in range(num_layers_mlp):
            image_model.add(Dense(num_hidden_nodes_mlp, kernel_initializer='uniform'))
            image_model.add(Activation(activation_mlp))
            image_model.add(Dropout(dropout))
            image_model.add(Dense(nb_classes, kernel_initializer='uniform'))
        image_model.add(Activation('softmax'))

        print(image_model.summary())

        #-------------------------------------------------------------------------------------------------
        # Language mode, LSTM party
        language_model = Sequential()
        language_model.add(LSTM(output_dim=num_hidden_nodes_lstm,
                                return_sequences=True, input_shape=(None, word2vec_dim)))

        for i in range(num_layers_lstm-2):
            language_model.add(LSTM(output_dim=num_hidden_nodes_lstm, return_sequences=True))
        language_model.add(LSTM(output_dim=num_hidden_nodes_lstm, return_sequences=False))

        print(language_model.summary())

        #-------------------------------------------------------------------------------------------------
        # Merging both models
        merged_output = concatenate([language_model.output, image_model.output])

        # Add fully connected layers
        for i in range(num_layers_mlp):

            if i == 0:
                x = merged_output

            x = Dense(num_hidden_nodes_mlp, init='uniform')(x)
            x = Activation('tanh')(x)
            x = Dropout(0.5)(x)

        x = Dense(upper_lim)(x)
        x = (Activation("softmax"))(x)


        #model_dump = model.to_json()
        #open(os.path.join(data_folder, "output/lstm_structure'  + '.json'"), 'w').write(model_dump)

        final_model = Model([language_model.input, image_model.input], x)

        final_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        print(final_model.summary())

        plot_model(final_model, to_file='./model.png')

        tboard = TensorBoard(log_dir='./', write_graph=True, write_grads=True, batch_size=batch_size, write_images=True)

        X_ques_batch = get_questions_tensor_timeseries(training_questions, nlp, batch_size)

        X_img_batch = get_images_matrix(images_train, id_map, img_features)

        Y_batch = get_answers_sum(answers_train, lbl)

        print(X_ques_batch.shape, X_img_batch.shape, Y_batch.shape)

        final_model.fit([X_ques_batch, X_img_batch], Y_batch, epochs=2, batch_size=256, verbose=2, callbacks=[tboard])

        final_model.save_weights(os.path.join(data_folder, "output/LSTM" + "_epoch_{}.hdf5".format("Hola_Hector")))

