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

class VQA_predict(object):
    """
    Evaluation class for the VQA
    """

    def prediction(self, data_folder, time_and_uuid, subset_size=1, model_type=2):
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
        test_questions = open(os.path.join(data_folder, "evaluate/ques_val.txt"), "rb").read().decode(
            'utf8').splitlines()
        test_questions_len = open(os.path.join(data_folder, "evaluate/ques_val_len.txt"), "rb").read().decode(
            'utf8').splitlines()
        test_answers = open(os.path.join(data_folder, "evaluate/answer_val.txt"), "rb").read().decode(
            'utf8').splitlines()
        test_images = open(os.path.join(data_folder, "evaluate/val_images_coco_id.txt"), "rb").read().decode(
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

        #VGG_LSTM
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
        test_questions_f, test_answers_f, test_images_f = freq_answers(test_questions, test_answers, test_images, upper_lim)

        test_questions_len, test_questions, test_answers, test_images = (list(t) for t in zip(*sorted(zip(test_questions_len,
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
        pk.dump(lbl, open(os.path.join(data_folder, "output/label_encoder_lstm.sav"),'wb'))

        timestep = len(nlp(subset_questions[-1]))

        # load_structure
        with open(structure, 'r') as json_file:
            model_prediction = model_from_json(json_file.read())

        # load model
        model_prediction.load_weights(weights)

        # summarize model.
        model_prediction.summary()

        # --------------------------------------------------------------------------------------------------------------

        """
        print("Getting questions batch")
        X_ques = get_questions_tensor_timeseries(subset_questions, nlp, timestep)

        print("Getting images batch")
        X_img = get_images_matrix_VGG(subset_images, data_folder, train_or_val='val')

        print("Get answers batch ")
        Y_answers = get_answers_matrix(subset_answers, lbl)
        """



        # --------------------------------------------------------------------------------------------------------------
        # PREDICTIONS
        """
        output = tf.keras.metrics.top_k_categorical_accuracy(Y_answers, prediction, k=10)

        with tf.Session() as sess:
            result = sess.run(output)

        print("Top k", result)
        """
        

        # This only works with Sequential
        #prediction = model_prediction.predict_classes([X_ques, X_img], verbose=1, batch_size=1)

        for ques_id, img_id, ans_id in zip(subset_questions, subset_images, subset_answers):
            # evaluate the model
            print(ques_id, img_id, ans_id)

            print("Getting questions batch")
            X_ques = get_questions_tensor_timeseries_predict(ques_id, nlp, timestep)

            print("Getting images batch")
            X_img = get_images_matrix_VGG(img_id, data_folder, train_or_val='val')

            print("Get answers batch ")
            Y_answers = get_answers_matrix_prediction(ans_id, lbl)

            print(X_ques.shape, X_img.shape, Y_answers.shape)

            prediction = model_prediction.predict([X_ques, X_img], verbose=1)
            y_classes = prediction.argmax(axis=-1)

            output = tf.keras.metrics.top_k_categorical_accuracy(Y_answers, prediction, k=10)

            with tf.Session() as sess:
                result = sess.run(output)

            print("Top k", result)


            top_values = [y_classes[i] for i in np.argsort(y_classes)[-5:]]
            print(top_values)

            imgFilename = 'COCO_' + 'val2014' + '_' + str(img_id).zfill(12) + '.jpg'

            I = io.imread(os.path.join(data_folder, 'Images/val2014/') + imgFilename)
            plt.title(ques_id, fontdict=None, loc='center', pad=None)

            plt.xlabel("Ground truth: " + str(ans_id) + " Pred: " + str(''.join(lbl.classes_[y_classes])))
            plt.imshow(I)
            fig1 = plt.gcf()
            plt.show()
            figure_name = (os.path.join(self.predictions_folder, '{}_{}.png'.format(unique_id, img_id)))
            fig1.savefig(figure_name)


