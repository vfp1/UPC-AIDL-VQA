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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import uuid

import json

class VQA_predict(object):
    """
    Evaluation class for the VQA
    """

    def prediction(self, data_folder, structure, weights, subset_size=4):
        """

        :param data_folder: the folder to the VQA data
        :param model: the model that we want to load, in hdf5 data type
        :param weights: the weights associated with that model
        :return: the final accuracy for the model
        """

        # Point to preprocessed data
        training_questions = open(os.path.join(data_folder, "evaluate/ques_val.txt"), "rb").read().decode(
            'utf8').splitlines()
        training_questions_len = open(os.path.join(data_folder, "evaluate/ques_val_len.txt"), "rb").read().decode(
            'utf8').splitlines()
        answers_train = open(os.path.join(data_folder, "evaluate/answer_val.txt"), "rb").read().decode(
            'utf8').splitlines()
        images_train = open(os.path.join(data_folder, "evaluate/val_images_coco_id.txt"), "rb").read().decode(
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

        """    
        # Random seed for constant random samples
        seed = 42
        random.seed(seed)
        """

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

        timestep = len(nlp(subset_questions[-1]))

        # load_structure
        structure_path = os.path.join(data_folder, structure)

        with open(structure_path, 'r') as json_file:
            model_prediction = model_from_json(json_file.read())

        # load model
        weights_path = os.path.join(data_folder, weights)
        model_prediction.load_weights(weights_path)

        # summarize model.
        model_prediction.summary()

        unique_id = str(uuid.uuid4())[:8]

        try:

            dir_tools = DirectoryTools()
            self.output_VGGLSTM_reports = dir_tools.folder_creator(input_folder=os.path.join(data_folder, 'output/VGG_LSTM/reports'))
            self.output_VGGLSTM_reports_uuid = dir_tools.folder_creator(input_folder=os.path.join(data_folder, 'output/VGG_LSTM/reports/{}'.format(unique_id)))

        except:

            print("Output_folder_not_created")

        print("Getting questions batch")
        X_ques = get_questions_tensor_timeseries(subset_questions, nlp, timestep)

        print("Getting images batch")
        X_img = get_images_matrix_VGG(images_train, subset_images, data_folder, train_or_val='val')

        prediction = model_prediction.predict([X_ques, X_img], verbose=1)
        y_classes = prediction.argmax(axis=-1)

        # This only works with Sequential
        #prediction = model_prediction.predict_classes([X_ques, X_img], verbose=1, batch_size=1)

        for ques_id, img_id, ans_id, pred in zip(range(len(subset_questions)), range(len(subset_images)), range(len(subset_answers)), y_classes):
            # evaluate the model
            print(subset_questions[ques_id], subset_images[img_id], subset_answers[ans_id], lbl.classes_[pred])

            print(img_id)

            imgFilename = 'COCO_' + 'val2014' + '_' + str(subset_images[int(img_id)]).zfill(12) + '.jpg'

            I = io.imread(os.path.join(data_folder, 'Images/val2014/') + imgFilename)
            plt.title(subset_questions[int(ques_id)], fontdict=None, loc='center', pad=None)
            plt.xlabel("Ground truth: " + subset_answers[int(ans_id)] + " Pred: " + lbl.classes_[pred])
            plt.imshow(I)
            fig1 = plt.gcf()
            plt.show()
            figure_name = (os.path.join(self.output_VGGLSTM_reports_uuid, '{}_{}.png'.format(unique_id, img_id)))
            fig1.savefig(figure_name)


