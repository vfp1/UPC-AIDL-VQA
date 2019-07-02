#!/usr/bin/env python
import matplotlib.pyplot as plt
import csv
import warnings

import datetime
import warnings
warnings.filterwarnings("ignore")
from random import shuffle, sample
import random
import pickle as pk
import gc

import os

import numpy as np
import pandas as pd
import scipy.io
from keras.models import Sequential, Model
from keras.layers import concatenate, multiply
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.merge import Concatenate
from keras.optimizers import RMSprop
from keras.utils import np_utils, generic_utils
from progressbar import Bar, ETA, Percentage, ProgressBar
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

import spacy
#from spacy.en import English

# Adapting to Google Cloud imports
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

from keras.utils import plot_model, Sequence
from keras.callbacks import TensorBoard, CSVLogger

from sklearn.model_selection import train_test_split

from vqaHelpers import VQA
import skimage.io as io

class Custom_Batch_Generator(Sequence):
    # Here we inherit the Sequence class from keras.utils
    """
    A custom Batch Generator to load the dataset from the HDD in
    batches to memory
    """

    def __init__(self, questions, images, answers, batch_size, lstm_timestep,
                 data_folder, nlp_load, lbl_load, tf_crop, img_standarization):
        """
        Here, we can feed parameters to our generator.
        :param questions: the preprocessed questions
        :param images: the preprocessed images
        :param answers: the preprocessed answers
        :param batch_size: the batch size
        :param lstm_timestep: the timestep of the LSTM timestep = len(nlp(subset_questions[-1]))
        :param data_folder: the data root folder (/aidl/VQA/data/, in Google Cloud
        :param nlp_load: the nlp processing to be loaded from VQA_train
        :param lbl_load: the lbl processing to be loaded from VQA_train
        :param val_coco_id_file: the file for the coco id
        """
        self.questions = questions
        self.images = images
        self.answers = answers
        self.batch_size = batch_size
        self.lstm_timestep = lstm_timestep
        self.data_folder = data_folder
        self.nlp_load = nlp_load
        self.lbl_load = lbl_load
        self.tf_crop = tf_crop
        self.img_standarization = img_standarization

    def __len__(self):
        """
        This function computes the number of batches that this generator is supposed to produce.
        So, we divide the number of total samples by the batch_size and return that value.
        :return:
        """
        print("Batches per epoch:", (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int))

        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):

        """
        Here, given the batch number idx you need to put together a list
        that consists of data batch and the ground-truth (GT).

        In our case, this is a tuple of [questions, images] and [answers]

        In __getitem__(self, idx) function you can decide what happens to your dataset when loaded in batches.
        Here, you can put your preprocessing steps as well.

        :param idx: the batch id
        :return:
        """
        batch_x_questions = self.questions[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_x_images = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_y_answers = self.answers[idx * self.batch_size: (idx + 1) * self.batch_size]

        print("Getting questions batch")
        X_ques_batch_fit = get_questions_tensor_timeseries(batch_x_questions, self.nlp_load, self.lstm_timestep)

        print("Getting images batch")
        X_img_batch_fit = get_images_matrix_VGG(batch_x_images, self.data_folder,
                                                train_or_val='train', standarization=self.img_standarization,
                                                tf_pad_resize=self.tf_crop)

        print("Get answers batch ")
        Y_batch_fit = get_answers_matrix(batch_y_answers, self.lbl_load)

        print(X_ques_batch_fit.shape, X_img_batch_fit.shape, Y_batch_fit.shape)

        return [X_ques_batch_fit, X_img_batch_fit], Y_batch_fit

class VQA_train(object):
    """
    The training of VQA
    """

    def train(self, unique_id, data_folder, model_type=1, num_epochs=4, subset_size=25000, subset=False,
              bsize=256, steps_per_epoch=20, keras_loss='categorical_crossentropy',
              keras_metrics='categorical_accuracy', learning_rate=0.01,
              optimizer='rmsprop', fine_tuned=True, test_size=0.20, vgg_frozen=4,
              lstm_hidden_nodes=512, lstm_num_layers=3, fc_hidden_nodes=1024, fc_num_layers=3,
              merge_method='concatenate', tf_crop_bool=False, image_standarization=True, vgg_finetuned_dropout=0.5,
              vgg_finetuned_activation='relu', merged_dropout_num=0.5, merged_activation='tanh',
              finetuned_batchnorm=True, merged_batchnorm=True):

        """
        Defines the training

        :param unique_id: the unique id for the experiment
        :param data_folder: the root data folder
        :param model_type: 1, MLP, LSTM; 2, VGG, LSTM
        :param num_epochs: the number of epochs
        :param subset_size: the subset size of VQA dataset, recommended 25000 by default
        :param subset: whether to subset the dataset or not
        :param bsize: the batch size, default at 256
        :param steps_per_epoch: the steps for each epoch
        :param keras_loss: the chosen keras loss
        :param keras_metrics: the chosen keras metrics
        :param learning_rate: the chosen learning rate
        :param optimizer: the chosen optimizer
        :param fine_tuned: whether to fine tune VGG or not
        :param test_size: the split test size, set to 80/20
        :param vgg_frozen: the number of frozen layers
        :param lstm_hidden_nodes: the LSTM hidden nodes, set to 512
        :param lstm_num_layers: the number of chosen LSTM layers
        :param fc_hidden_nodes: the number of FC hidden nodes after model merges, set to 1024
        :param fc_num_layers: the number of FC layers (DENSE)
        :param merge_method: the chosen merge method, either concatenate or dot
        :param tf_crop_bool: True/False cropping the images with tensorflow (True) or scikit image (False)
        :param image_standarization: whether to do image scaling for zero mean and unit variance
        :param vgg_finetuned_dropout: the dropout for the fine tuned VGG
        :param vgg_finetuned_activation: the activation for the fine tuned VGG
        :param merged_dropout_num: the dropout for the merged part
        :param merged_activation: the activation function for the merged part
        :param finetuned_batchnorm: the batchnorm for the fine tuned part
        :param merged_batchnorm: the batchnorm for the merged part

        :return: the VQA train
        """

        # Setting Hyperparameters
        batch_size = bsize
        img_dim = 4096  # the image dimensions for the MLP and the output of the FCN
        word2vec_dim = 96
        #max_len = 30 # Required only when using Fixed-Length Padding

        num_hidden_nodes_mlp = fc_hidden_nodes
        num_hidden_nodes_lstm = lstm_hidden_nodes
        num_layers_mlp = fc_num_layers
        num_layers_lstm = lstm_num_layers
        dropout = merged_dropout_num
        activation_mlp = 'tanh'

        num_epochs = num_epochs
        #log_interval = 15

        # Point to preprocessed data
        training_questions = open(os.path.join(data_folder, "preprocessed/ques_val.txt"),"rb").read().decode('utf8').splitlines()
        training_questions_len = open(os.path.join(data_folder, "preprocessed/ques_val_len.txt"),"rb").read().decode('utf8').splitlines()
        answers_train = open(os.path.join(data_folder, "preprocessed/answer_val.txt"),"rb").read().decode('utf8').splitlines()
        images_train = open(os.path.join(data_folder, "preprocessed/val_images_coco_id.txt"),"rb").read().decode('utf8').splitlines()
        #img_ids = open(os.path.join(data_folder,'preprocessed/coco_vgg_IDMap.txt')).read().splitlines()
        #img_ids = open(os.path.join(data_folder,'preprocessed/val_images_coco_id.txt')).read().splitlines()
        vgg_path = os.path.join(data_folder, "vgg_weights/vgg16_weights.h5")

        # Load english dictionary
        try:
            nlp = spacy.load("en_core_web_md")
        except:
            nlp = spacy.load("en_core_web_sm")
        print("Loaded Word2Vec for question encoding")
        

        # Load VGG weights (only for MLP, for VGG are loaded later)
        try:
            vgg_features = scipy.io.loadmat(vgg_path)
            img_features = vgg_features['feats']
        except:
            pass

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

        if subset is True:
            # Creating subset
            subset_questions = []
            subset_answers = []
            subset_images = []

            """
            This below is the total sample size that will be created. It needs to be at least bigger than
            1000 samples, since we have 1000 possible types of questions
            """

            sample_size = subset_size

            # Random seed for constant random samples
            seed=42
            random.seed(seed)

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
            print("Sanity check full subset")
            print("Are all of the giraffes standing up straight? no 73726")
            print(subset_questions[1], subset_answers[1], subset_images[1])

        elif subset is False:
            print("TRAINING WITH NO SUBSET")

            # Creating subset
            subset_questions = training_questions
            subset_answers = answers_train
            subset_images = images_train

        print("-----------------------------------------------------------------------")
        print("TRAIN/TEST SPLIT:")

        # Using random state = 42 to get always same train/test splits
        subset_questions_train, subset_questions_val, subset_images_train, subset_images_val, subset_answers_train, subset_answers_val  = train_test_split(subset_questions,
                                                                                                                                                           subset_images,
                                                                                                                                                           subset_answers,
                                                                                                                                                           test_size=test_size,
                                                                                                                                                           random_state=42,
                                                                                                                                                           shuffle=True)

        print("Lenght train:", len(subset_questions_train), len(subset_images_train), len(subset_answers_train))
        print("Lenght validation:", len(subset_questions_val), len(subset_images_val), len(subset_answers_val))
        print("-----------------------------------------------------------------------")
        print("Sanity check train test")

        print("Is there traffic? yes 540899")
        print(subset_questions_train[1], subset_answers_train[1], subset_images_train[1])
        print("What color are the words? yellow 172537")
        print(subset_questions_val[1], subset_answers_val[1], subset_images_val[1])

        print("-----------------------------------------------------------------------")
        print("ENCODING ANSWERS")

        # Encoding answers
        lbl = LabelEncoder()
        lbl.fit(answers_train)
        nb_classes = len(list(lbl.classes_))
        print("Number of classes:", nb_classes)
        pk.dump(lbl, open(os.path.join(data_folder, "output/label_encoder_lstm.sav"),'wb'))

        # -------------------------------------------------------------------------------------------------
        # DECIDE MODEL TYPE: MPL_LSTM or VGG_LSTM

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

            img_ids = open(os.path.join(data_folder,'preprocessed/coco_vgg_IDMap.txt')).read().splitlines()

            id_map = dict()

            for ids in img_ids:
                id_split = ids.split()
                id_map[id_split[0]] = int(id_split[1])

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
            
            # This is the VGG model 
            
            image_model = Sequential()

            if fine_tuned is False:
                image_model = VGG().VGG_16()
                VGG_weights = "FALSE"

            elif fine_tuned is True:
                image_model = VGG().VGG_16_pretrained(frozen_layers=vgg_frozen,
                                                      vgg_fine_tune_dropout=vgg_finetuned_dropout,
                                                      vgg_fine_tune_activation=vgg_finetuned_activation,
                                                      batch_norm=finetuned_batchnorm)
                VGG_weights = "TRUE"

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
            # MERGING BOTH MODELS AND SETTING OPTIMIZER AND COMPILER

            if merge_method == 'concatenate':
                merged_output = concatenate([language_model.output, image_model.output])
            elif merge_method == 'multiply':
                merged_output = multiply([language_model.output, image_model.output])

            # Add fully connected layers
            for i in range(num_layers_mlp):

                if i == 0:
                    x = merged_output

                x = Dense(num_hidden_nodes_mlp, init='uniform')(x)
                x = Activation(merged_activation)(x)
                if merged_batchnorm is True:
                    x = BatchNormalization()(x)
                elif merged_batchnorm is False:
                    pass
                x = Dropout(merged_dropout_num)(x)

            x = Dense(upper_lim)(x)

            # Change last activation depending on loss function
            if keras_loss == 'categorical_hinge':
                x = (Activation("linear"))(x)
            else:
                x = (Activation("softmax"))(x)

            final_model = Model([language_model.input, image_model.input], x)

            if optimizer == 'rmsprop':

                opt = RMSprop(lr=learning_rate)

            else:

                raise ValueError("Optimizer not loaded")

            final_model.compile(loss=keras_loss, optimizer=opt, metrics=[keras_metrics, 'accuracy'])

            print(final_model.summary())

            # ------------------------------------------------------------------------------------------------------------------
            # EXPERIMENT REPORTING SETTINGS

            time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

            try:

                dir_tools = DirectoryTools()

                self.output_VGGLSTM_folder = dir_tools.folder_creator(
                    input_folder=os.path.join(data_folder, 'output/VGG_LSTM'))

                self.output_VGGLSTM_reports = dir_tools.folder_creator(
                    input_folder=os.path.join(data_folder, 'output/VGG_LSTM/reports'))

                self.output_VGGLSTM_reports_uuid = dir_tools.folder_creator(
                    input_folder=os.path.join(data_folder, 'output/VGG_LSTM/reports/{}_{}'.format(time_string, unique_id)))

                csv_file_name = os.path.join(self.output_VGGLSTM_reports_uuid, "parameters_{}.csv".format(unique_id))

                with open(csv_file_name, mode='w', newline='') as csv_file:
                    filewriter = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    filewriter.writerow(['Time', '{}'.format(time_string)])
                    filewriter.writerow(['Folder name', '{}'.format(unique_id)])
                    filewriter.writerow(['Epochs', '{}'.format(num_epochs)])
                    filewriter.writerow(['Batch size', '{}'.format(batch_size)])
                    if subset is True:
                        filewriter.writerow(['Subset', '{}'.format(sample_size)])
                    elif subset is False:
                        filewriter.writerow(['Batch size', '{}'.format(len(sample_size))])
                    filewriter.writerow(['Loss', '{}'.format(keras_loss)])
                    filewriter.writerow(['VGG weights', '{}'.format(VGG_weights)])
                    filewriter.writerow(['VGG frozen layers', '{}'.format(vgg_frozen)])
                    filewriter.writerow(['Metrics', '{}'.format(keras_metrics)])
                    filewriter.writerow(['Optimizer', '{}'.format(optimizer)])
                    filewriter.writerow(['Learning rate', '{}'.format(learning_rate)])
                    filewriter.writerow(['Test size', '{}'.format(test_size)])
                    filewriter.writerow(['Word2vec dimension', '{}'.format(word2vec_dim)])
                    filewriter.writerow(['Hidden nodes LSTM', '{}'.format(num_hidden_nodes_lstm)])
                    filewriter.writerow(['Hidden layers LSTM', '{}'.format(num_layers_lstm)])
                    filewriter.writerow(['Hidden nodes FC', '{}'.format(num_hidden_nodes_mlp)])
                    filewriter.writerow(['Number layers FC', '{}'.format(num_layers_mlp)])
                    filewriter.writerow(['Merge methods', '{}'.format(merge_method)])
                    filewriter.writerow(['Image TF Crop', '{}'.format(tf_crop_bool)])
                    filewriter.writerow(['Image Standarization', '{}'.format(image_standarization)])
                    filewriter.writerow(['Dropout in the fine tuned VGG', '{}'.format(vgg_finetuned_dropout)])
                    filewriter.writerow(['Activation in the fine tuned VGG', '{}'.format(vgg_finetuned_activation)])
                    filewriter.writerow(['Dropout in the merged model', '{}'.format(merged_dropout_num)])
                    filewriter.writerow(['Activation in the merged model', '{}'.format(merged_activation)])
                    filewriter.writerow(['Batch normalization fine tuned', '{}'.format(finetuned_batchnorm)])
                    filewriter.writerow(['Batch normalization merged model', '{}'.format(merged_batchnorm)])


            except:

                warnings.warn("Experiment csv not generated")

                pass

            # ------------------------------------------------------------------------------------------------------------------
            # SAVE MODEL STRUCTURE

            try:

                # Save model structure

                model_dump = final_model.to_json()
                with open(
                        os.path.join(self.output_VGGLSTM_reports_uuid, 'vgg_lstm_structure_{}.json'.format(unique_id)),
                        'w') as dump:
                    dump.write(model_dump)

            except:

                warnings.warn("Model structure generated")

                pass

            # ------------------------------------------------------------------------------------------------------------------
            # SAVE MODEL PLOT
            try:

                plot_model(final_model, to_file=os.path.join(self.output_VGGLSTM_reports_uuid, 'VGG_LSTM_{}.png'.format(unique_id)))

            except:

                warnings.warn("Model plot not generated")

                pass

            # ------------------------------------------------------------------------------------------------------------------
            # TENSORBOARD ACTIVATION
            try:
                # Tensorboard capabilities
                dir_tools = DirectoryTools()

                self.output_VGGLSTM_tboard = dir_tools.folder_creator(
                    input_folder=os.path.join(data_folder, 'output/VGG_LSTM/tboard_logs'))


                path_file = os.path.join(self.output_VGGLSTM_tboard, "{}_{}".format(time_string, unique_id))

                tboard = TensorBoard(log_dir=path_file, write_graph=True, write_grads=True,
                                     batch_size=batch_size, write_images=True)

            except:

                warnings.warn("Tensorboard not generated")

                pass

            # ------------------------------------------------------------------------------------------------------------------
            # CSV HISTORY LOGGER

            try:
                # Save history to CSV
                csv_logger = CSVLogger(os.path.join(self.output_VGGLSTM_reports_uuid, "history_{}.csv".format(unique_id)), append=True, separator=';')

            except:

                warnings.warn("CSV Logger not generated")

                pass

            #-----------------------------------------------------------------------
            # Preparing train

            print("-----------------------------------------------------------------------")
            print("GENERATING THE TRAIN BATCH GENERATOR")

            timestep = len(nlp(subset_questions_train[-1]))

            train_batch_generator = Custom_Batch_Generator(questions=subset_questions_train, images=subset_images_train,
                                                              answers=subset_answers_train, batch_size=batch_size,
                                                              lstm_timestep=timestep, data_folder=data_folder,
                                                              nlp_load=nlp, lbl_load=lbl, tf_crop=tf_crop_bool, img_standarization=image_standarization)

            print("-----------------------------------------------------------------------")
            print("GENERATING THE VALIDATION BATCH GENERATOR")

            timestep = len(nlp(subset_questions_val[-1]))

            validation_batch_generator = Custom_Batch_Generator(questions=subset_questions_val, images=subset_images_val,
                                                              answers=subset_answers_val, batch_size=batch_size,
                                                              lstm_timestep=timestep, data_folder=data_folder,
                                                              nlp_load=nlp, lbl_load=lbl, tf_crop=tf_crop_bool, img_standarization=image_standarization)

            print("-----------------------------------------------------------------------")
            print("TRAINING")

            # Deploying in Google Cloud (Linux VM)

            #The steps per epoch are the int(sample_size // batch_size). However, it can get too heavy, so I will leave 
            #a multiple number of the batch size
         

            try:
                history = final_model.fit_generator(generator=train_batch_generator,
                                                    steps_per_epoch=steps_per_epoch,
                                                    epochs=num_epochs,
                                                    verbose=2,
                                                    validation_data=validation_batch_generator,
                                                    validation_steps=int(steps_per_epoch // 4),
                                                    callbacks=[tboard, csv_logger])

                # list all data in history
                print(history.history.keys())
                # summarize history for accuracy
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                fig1 = plt.gcf()
                figure_name = (os.path.join(self.output_VGGLSTM_reports_uuid, '{}_{}.png'.format("accuracy", unique_id)))
                fig1.savefig(figure_name)

                # clear model
                plt.clf()

                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                fig2 = plt.gcf()
                figure_name = (os.path.join(self.output_VGGLSTM_reports_uuid, '{}_{}.png'.format("loss", unique_id)))
                fig2.savefig(figure_name)

                # SAVE WEIGHTS

                final_model.save_weights(os.path.join(self.output_VGGLSTM_reports_uuid, "VGG_LSTM_WEIGHTS_uuid_{}.hdf5".format(unique_id)))

            except:

                history = final_model.fit_generator(generator=train_batch_generator,
                                                    steps_per_epoch=steps_per_epoch,
                                                    epochs=num_epochs,
                                                    verbose=2,
                                                    validation_data=validation_batch_generator,
                                                    validation_steps=int(steps_per_epoch // 4),
                                                    callbacks=[csv_logger])

                # list all data in history
                print(history.history.keys())
                # summarize history for accuracy
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                fig1 = plt.gcf()
                figure_name = (
                    os.path.join(self.output_VGGLSTM_reports_uuid, '{}_{}.png'.format("accuracy", unique_id)))
                fig1.savefig(figure_name)

                # clear model
                plt.clf()

                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                fig2 = plt.gcf()
                figure_name = (
                    os.path.join(self.output_VGGLSTM_reports_uuid, '{}_{}.png'.format("loss", unique_id)))
                fig2.savefig(figure_name)

                # SAVE WEIGHTS

                final_model.save_weights(os.path.join(self.output_VGGLSTM_reports_uuid, "VGG_LSTM_WEIGHTS_uuid_{}.hdf5".format(unique_id)))

        print("")
        print("CONGRATULATIONS! TRAIN COMPLETED")


