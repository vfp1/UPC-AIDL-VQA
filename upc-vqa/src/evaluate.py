#!/usr/bin/env python

# load and evaluate a saved model
import os
from keras.models import load_model

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
import picke as pk

class VQA_evaluate(object):
    """
    Evaluation class for the VQA
    """

    def evaluation(self, data_folder, weights, subset_size):
        """

        :param data_folder: the folder to the VQA data
        :param model: the model that we want to load, in hdf5 data type
        :param weights: the weights associated with that model
        :return: the final accuracy for the model
        """

        # Point to preprocessed data
        training_questions = open(os.path.join(data_folder, "preprocessed/ques_val.txt"), "rb").read().decode(
            'utf8').splitlines()
        training_questions_len = open(os.path.join(data_folder, "preprocessed/ques_val_len.txt"), "rb").read().decode(
            'utf8').splitlines()
        answers_train = open(os.path.join(data_folder, "preprocessed/answer_val.txt"), "rb").read().decode(
            'utf8').splitlines()
        images_train = open(os.path.join(data_folder, "preprocessed/val_images_coco_id.txt"), "rb").read().decode(
            'utf8').splitlines()


        # Number of most frequently occurring answers in COCOVQA (Covering >80% of the total data)
        upper_lim = 1000

        # Loading most frequent occurring answers
        training_questions, answers_train, images_train = freq_answers(training_questions,
                                                                       answers_train, images_train, upper_lim)


        # Creating subset
        subset_questions = []
        subset_answers = []
        subset_images = []

        """
        This below is the total sample size that will be created. It needs to be at least bigger than
        1000 samples, since we have 1000 possible types of questions
        """

        sample_size = subset_size

        for index in sorted(random.sample(range(len(images_train)), sample_size)):
            subset_questions.append(training_questions[index])
            subset_answers.append(answers_train[index])
            subset_images.append(images_train[index])

        # Load english dictionary
        try:
            nlp = spacy.load("en_core_web_md")
        except:
            nlp = spacy.load("en_core_web_sm")
        print("Loaded Word2Vec for question encoding")

        # Encoding answers
        lbl = LabelEncoder()
        lbl.fit(answers_train)
        nb_classes = len(list(lbl.classes_))
        print("Number of classes:", nb_classes)
        pk.dump(lbl, open(os.path.join(data_folder, "output/label_encoder_lstm.sav"),'wb'))


        # Getting data to tensors

        timestep = len(nlp(subset_questions[-1]))

        print("Getting questions batch")
        X_ques = get_questions_tensor_timeseries(subset_questions, nlp, timestep)

        print("Getting images batch")
        X_img = get_images_matrix_VGG(images_train, subset_images, data_folder)

        print("Get answers batch ")
        Y_answers = get_answers_matrix(subset_answers, lbl)


        # load model
        model_evaluation = load_model(weights)

        # summarize model.
        model_evaluation.summary()


        # evaluate the model
        score = model_evaluation.evaluate([X_ques, X_img], Y_answers, verbose=1)
        print("%s: %.2f%%" % (model_evaluation.metrics_names[1], score[1] * 100))