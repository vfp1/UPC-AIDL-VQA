#!/usr/bin/env python

import datetime
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
try:
    from src.features import *
    from src.utils import *
    from src.vqa_data_prep import *
    from src.models import *
except:
    from features import *
    from utils import *
    from vqa_data_prep import *
    from models import *


import graphviz
import pydot_ng as pydot
pydot.find_graphviz()

from keras.utils import plot_model
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

from vqaHelpers import VQA
import skimage.io as io


class VQA_train(object):
    """

    """

    def train(self, data_folder, model_type=1):
        """

        :param data_folder: the root data folder
        :param model_type: 1, MLP, LSTM; 2, VGG, LSTM
        :return:
        """

        # Point to preprocessed data
        training_questions = open(os.path.join(data_folder, "preprocessed/ques_val.txt"),"rb").read().decode('utf8').splitlines()
        training_questions_len = open(os.path.join(data_folder, "preprocessed/ques_val_len.txt"),"rb").read().decode('utf8').splitlines()
        answers_train = open(os.path.join(data_folder, "preprocessed/answer_val.txt"),"rb").read().decode('utf8').splitlines()
        images_train = open(os.path.join(data_folder, "preprocessed/val_images_coco_id.txt"),"rb").read().decode('utf8').splitlines()
        #img_ids = open(os.path.join(data_folder,'preprocessed/coco_vgg_IDMap.txt')).read().splitlines()
        #img_ids = open(os.path.join(data_folder,'preprocessed/val_images_coco_id.txt')).read().splitlines()
        vgg_path = os.path.join(data_folder, "vgg_weights/vgg_feats.mat")

        # Load english dictionary
        try:
            nlp = spacy.load("en_core_web_md")
        except:
            nlp = spacy.load("en_core_web_sm")
        print ("Loaded WordVec")
        

        # Load VGG weights

        vgg_features = scipy.io.loadmat(vgg_path)
        img_features = vgg_features['feats']

        #id_map = dict()
        print("Loaded VGG Weights")


        # Number of most frequently occurring answers in COCOVQA (Covering >80% of the total data)
        upper_lim = 1000

        # Loading most frequent occurring answers
        training_questions, answers_train, images_train = freq_answers(training_questions,
                                                               answers_train, images_train, upper_lim)

        training_questions_len, training_questions, answers_train, images_train = (list(t) for t in zip(*sorted(zip(training_questions_len,
                                                                                                          training_questions, answers_train,
                                                                                                          images_train))))
        # Sanity check
        print("-----------------------------------------------------------------------")
        print("Before subset:")
        print("Lenght training questions:", len(training_questions))
        print("Lenght answers", len(answers_train))
        print("Lenght number images", len(images_train))

        # Creating subset
        import random

        subset_questions = []
        subset_answers = []
        subset_images = []

        """
        This below is the total sample size that will be created. It needs to be at least bigger than
        1000 samples, since we have 1000 possible types of questions
        """

        sample_size = 4

        for index in sorted(random.sample(range(len(images_train)), sample_size)):
            subset_questions.append(training_questions[index])
            subset_answers.append(answers_train[index])
            subset_images.append(images_train[index])


        # Sanity check
        print("-----------------------------------------------------------------------")
        print("A sanity check: Lenght training questions:", len(subset_questions))
        print("Lenght training answers:", len(subset_answers))
        print("Lenght number images", len(subset_images))
        print("-----------------------------------------------------------------------")
        print("Sanity check")
        random_id = random.sample(range(len(subset_images)), 1)
        print(subset_questions[random_id[0]], subset_answers[random_id[0]], subset_images[random_id[0]])
        print("-----------------------------------------------------------------------")

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

        num_epochs = 2
        #log_interval = 15

        """
        This was used to map COCO, we have a subset of it
        
        for ids in img_ids:
            id_split = ids.split()
            id_map[id_split[0]] = int(id_split[1])
        """
        # -------------------------------------------------------------------------------------------------
        # DECIDE MODEL TYPE: MPL_LSTM or VGG_LSTM

        #TODO: fix break at id_map
        if model_type == 1:
            print("USING MLP LSTM model")

            #-------------------------------------------------------------------------------------------------
            # Image model, a very simple MLP
            image_model = Sequential()
            image_model.add(Dense(num_hidden_nodes_mlp, input_dim=img_dim, kernel_initializer='uniform'))
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

            final_model = Model([language_model.input, image_model.input], x)

            final_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

            print(final_model.summary())

            try:

                dir_tools = DirectoryTools()
                self.output_MLPLSTM_folder = dir_tools.folder_creator(input_folder=os.path.join(data_folder, 'output/MLP_LSTM'))

                try:

                    model_dump = final_model.to_json()
                    with open(os.path.join(self.output_MLPLSTM_folder, 'mlp_lstm_structure.json'), 'w') as dump:
                        dump.write(model_dump)

                except:

                    pass

                plot_model(final_model, to_file= os.path.join(self.output_MLPLSTM_folder, './model.png'))

                epoch_string = 'EPOCH_{}-'.format(num_epochs)
                batch_string = 'BSIZE_{}-'.format(batch_size)
                subset_string = 'SUBSET_{}-'.format(sample_size)

                # Start the time string
                time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

                log_string = time_string + epoch_string + batch_string + subset_string

                path_file = os.path.join(self.output_MLPLSTM_folder, "{}".format(log_string))

                tboard = TensorBoard(log_dir=path_file, write_graph=True, write_grads=True,
                                     batch_size=batch_size, write_images=True)

            except:

                pass

            # This is the timestep of the NLP
            timestep = len(nlp(subset_questions[-1]))

            print("Getting questions")
            X_ques_batch_fit = get_questions_tensor_timeseries(subset_questions, nlp, timestep)

            print("Getting images")
            X_img_batch_fit = get_images_matrix(subset_images, id_map, img_features)

            print("Get answers")
            Y_batch_fit = get_answers_sum(subset_answers, lbl)

            print("Questions, Images, Answers")
            print(X_ques_batch_fit.shape, X_img_batch_fit.shape, Y_batch_fit.shape)


            print("-----------------------------------------------------------------------")
            print("TRAINING")

            try:

                final_model.fit([X_ques_batch_fit, X_img_batch_fit], Y_batch_fit, epochs=num_epochs, batch_size=batch_size, verbose=2,
                                callbacks=[tboard])
                final_model.save_weights(os.path.join(self.output_MLPLSTM_folder, "LSTM" + "_epoch_{}.hdf5".format("FINAL")))

            except:

                final_model.fit([X_ques_batch_fit, X_img_batch_fit], Y_batch_fit, epochs=num_epochs, batch_size=batch_size, verbose=2)
                final_model.save_weights(os.path.join(self.output_MLPLSTM_folder, "LSTM" + "_epoch_{}.hdf5".format("FINAL")))

        elif model_type == 2:

            print("USING VGG LSTM model")

            # -------------------------------------------------------------------------------------------------
            # Image model
            image_model = Sequential()
            #TODO: fix h5py load issue OSError: Unable to open file (file signature not found)
            try:
                image_model = VGG().VGG_16(weights_path=vgg_path)
            except:
                image_model = VGG().VGG_16()
                #image_model.add(Reshape((img_dim,), input_shape=(img_dim,)))

            print(image_model.summary())

            # -------------------------------------------------------------------------------------------------
            # Language mode, LSTM party
            language_model = Sequential()
            language_model.add(LSTM(output_dim=num_hidden_nodes_lstm,
                                    return_sequences=True, input_shape=(None, word2vec_dim)))

            for i in range(num_layers_lstm - 2):
                language_model.add(LSTM(output_dim=num_hidden_nodes_lstm, return_sequences=True))
            language_model.add(LSTM(output_dim=num_hidden_nodes_lstm, return_sequences=False))

            print(language_model.summary())

            # -------------------------------------------------------------------------------------------------
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


            final_model = Model([language_model.input, image_model.input], x)

            final_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

            print(final_model.summary())

            # ------------------------------------------------------------
            # Loading reporting capabilities
            try:

                dir_tools = DirectoryTools()
                self.output_VGGLSTM_folder = dir_tools.folder_creator(
                    input_folder=os.path.join(data_folder, 'output/VGG_LSTM'))

                try:

                    model_dump = final_model.to_json()
                    with open(os.path.join(self.output_VGGLSTM_folder, 'vgg_lstm_structure.json'), 'w') as dump:
                        dump.write(model_dump)

                except:

                    pass

                plot_model(final_model, to_file=os.path.join(self.output_VGGLSTM_folder, 'VGG_LSTM.png'))

                epoch_string = 'EPOCH_{}-'.format(num_epochs)
                batch_string = 'BSIZE_{}-'.format(batch_size)
                subset_string = 'SUBSET_{}-'.format(sample_size)

                # Start the time string
                time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

                log_string = time_string + epoch_string + batch_string + subset_string

                path_file = os.path.join(self.output_VGGLSTM_folder, "{}".format(log_string))

                tboard = TensorBoard(log_dir=path_file, write_graph=True, write_grads=True,
                                     batch_size=batch_size, write_images=True)

            except:

                pass

            #-----------------------------------------------------------------------
            # Preparing train

            # This is the timestep of the NLP
            timestep = len(nlp(subset_questions[-1]))

            print("Getting questions")
            X_ques_batch_fit = get_questions_tensor_timeseries(subset_questions, nlp, timestep)

            print("Getting images")
            X_img_batch_fit = get_images_matrix_VGG(subset_images, data_folder)
            #X_img_batch_fit_reshape = np.reshape(X_img_batch_fit, (-1, sample_size, img_dim, img_dim))

            print("Get answers")
            Y_batch_fit = get_answers_sum(subset_answers, lbl)

            print("Questions, Images, Answers")
            print(X_ques_batch_fit.shape, X_img_batch_fit.shape, Y_batch_fit.shape)

            print("-----------------------------------------------------------------------")
            print("TRAINING")

            try:

                final_model.fit([X_ques_batch_fit, X_img_batch_fit], Y_batch_fit, epochs=num_epochs,
                                batch_size=batch_size, verbose=2,
                                callbacks=[tboard])
                final_model.save_weights(
                    os.path.join(self.output_VGGLSTM_folder, "VGG_LSTM" + "_epoch_{}.hdf5".format("FINAL")))

            except:

                final_model.fit([X_ques_batch_fit, X_img_batch_fit], Y_batch_fit, epochs=num_epochs,
                                batch_size=batch_size, verbose=2)
                final_model.save_weights(
                    os.path.join(self.output_VGGLSTM_folder, "VGG_LSTM" + "_epoch_{}.hdf5".format("FINAL")))
