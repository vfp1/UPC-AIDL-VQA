#!/usr/bin/env python

# load and evaluate a saved model
import os
from keras.models import model_from_json

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

import random

from sklearn.preprocessing import LabelEncoder
import spacy
import pickle as pk

from keras.utils import plot_model, Sequence

import matplotlib.pyplot as plt

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

class VQA_evaluate(object):
    """
    Evaluation class for the VQA
    """

    def evaluate(self, data_folder, time_and_uuid, subset_size=1, model_type=2):
        """
        :param data_folder: the folder to the VQA data
        :param time_and_uuid: the weights associated with that model
        :param subset_size: the amount of predicts to do
        :param model_type: the type of model, 1= MLP_LSTM, 2=VGG_LSTM
        :return: the final accuracy for the model
        """

        # --------------------------------------------------------------------------------------------------------------
        # PREPARING PATHS

        # Point to preprocessed data
        test_questions = open(os.path.join(data_folder, "preprocessed/ques_val.txt"), "rb").read().decode(
            'utf8').splitlines()
        test_questions_len = open(os.path.join(data_folder, "preprocessed/ques_val_len.txt"), "rb").read().decode(
            'utf8').splitlines()
        test_answers = open(os.path.join(data_folder, "preprocessed/answer_val.txt"), "rb").read().decode(
            'utf8').splitlines()
        test_images = open(os.path.join(data_folder, "preprocessed/val_images_coco_id.txt"), "rb").read().decode(
            'utf8').splitlines()

        # Point to time_and_uuid sources
        global unique_id
        global weights
        global structure

        # MLP_LSTM
        if model_type == 1:

            results_folder = os.path.join(data_folder, 'output/VGG_LSTM/reports/{}'.format(time_and_uuid))

            unique_id = time_and_uuid.split("_", 1)[1]

            weights = os.path.join(results_folder, 'MLP_LSTM_WEIGHTS_uuid_{}.hdf5'.format(unique_id))

            structure = os.path.join(results_folder, 'mlp_lstm_structure_{}.json'.format(unique_id))

            dir_tools = DirectoryTools()
            self.predictions_folder = dir_tools.folder_creator(input_folder=os.path.join(results_folder, 'predictions'))

        # VGG_LSTM
        elif model_type == 2:

            results_folder = os.path.join(data_folder, 'output/VGG_LSTM/reports/{}'.format(time_and_uuid))

            unique_id = time_and_uuid.split("_", 1)[1]

            weights = os.path.join(results_folder, 'VGG_LSTM_WEIGHTS_uuid_{}.hdf5'.format(unique_id))

            structure = os.path.join(results_folder, 'vgg_lstm_structure_{}.json'.format(unique_id))

            dir_tools = DirectoryTools()
            self.predictions_folder = dir_tools.folder_creator(input_folder=os.path.join(results_folder, 'predictions'))

        # --------------------------------------------------------------------------------------------------------------
        # SETTING SUBSET

        # Number of most frequently occurring answers in COCOVQA (Covering >80% of the total data)
        upper_lim = 1000

        # Loading most frequent occurring answers
        test_questions_f, test_answers_f, test_images_f = freq_answers(test_questions, test_answers, test_images,
                                                                       upper_lim)

        test_questions_len, test_questions, test_answers, test_images = (list(t) for t in
                                                                         zip(*sorted(zip(test_questions_len,
                                                                                         test_questions_f,
                                                                                         test_answers_f,
                                                                                         test_images_f))))

        # Creating subset
        subset_questions = []
        subset_answers = []
        subset_images = []

        """
        This below is the total sample size that will be created. It needs to be at least bigger than
        1000 samples, since we have 1000 possible types of questions
        """

        sample_size = subset_size

        """    
        # Random seed for constant random samples
        seed = 42
        random.seed(seed)
        """

        for index in sorted(random.sample(range(len(test_images)), sample_size)):
            subset_questions.append(test_questions[index])
            subset_answers.append(test_answers[index])
            subset_images.append(test_images[index])

        # --------------------------------------------------------------------------------------------------------------
        # LOADING DICTIONARIES, STRUCTURES AND WEIGHTS

        # Load english dictionary
        try:
            nlp = spacy.load("en_core_web_md")
        except:
            nlp = spacy.load("en_core_web_sm")
        print("Loaded Word2Vec for question encoding")

        # Encoding answers
        lbl = LabelEncoder()
        lbl.fit(test_answers)
        nb_classes = len(list(lbl.classes_))
        print("Number of classes:", nb_classes)
        pk.dump(lbl, open(os.path.join(data_folder, "output/label_encoder_lstm.sav"), 'wb'))

        timestep = len(nlp(subset_questions[-1]))

        # load_structure
        with open(structure, 'r') as json_file:
            model_evaluation = model_from_json(json_file.read())

        # load model
        model_evaluation.load_weights(weights)

        # summarize model.
        model_evaluation.summary()

        # compile for evaluation
        model_evaluation.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # --------------------------------------------------------------------------------------------------------------
        evaluate_batch_generator = Custom_Batch_Generator(questions=subset_questions, images=subset_images,
                                                       answers=subset_images, batch_size=10,
                                                       lstm_timestep=timestep, data_folder=data_folder,
                                                       nlp_load=nlp, lbl_load=lbl, tf_crop=False,
                                                       img_standarization=True)

        evaluate = model_evaluation.evaluate_generator(evaluate_batch_generator, steps=None, max_queue_size=10, workers=1,
                           use_multiprocessing=False, verbose=0)


        print(evaluate)

        # --------------------------------------------------------------------------------------------------------------
