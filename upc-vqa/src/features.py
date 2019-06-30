import warnings
warnings.filterwarnings('ignore')

import numpy as np
from keras.utils import np_utils

import vqaHelpers
import random
import skimage.io as io
import os
from skimage.transform import resize

import tensorflow as tf
from tensorflow.python.keras import backend as K

from tqdm import tqdm

"""Gets the 4096-dimensional CNN features for the given COCO
	VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)
	Ouput:
	A numpy matrix of size (nb_samples, nb_dimensions)"""

def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
    assert not isinstance(img_coco_ids, str)
    nb_samples = len(img_coco_ids)
    nb_dimensions = VGGfeatures.shape[0]
    nb_dimensions_2 = VGGfeatures.shape[1]
    print("Dimensions", nb_dimensions, nb_dimensions_2)
    image_matrix = np.zeros((nb_samples, nb_dimensions))
    for j in range(len(img_coco_ids)):
        image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]

    return image_matrix

"""Gets the 4096-dimensional CNN features for the given COCO
	VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)
	Ouput:
	A numpy matrix of size (nb_samples, nb_dimensions)"""

def get_images_matrix_VGG(img_coco_subset, img_coco_batch, data_path, train_or_val='train', tf_pad_resize=True):

    #assert not isinstance(img_coco_ids, str)

    image_matrix = []

    # Using scikit image resize

    if tf_pad_resize is False:

        if train_or_val == 'val':

            for index, image_id in tqdm(zip(range(len(img_coco_batch)), img_coco_batch), total=len(img_coco_batch)):

                imgFilename = 'COCO_' + 'val2014' + '_' + str(img_coco_subset[int(index)]).zfill(12) + '.jpg'

                I = io.imread(os.path.join(data_path, 'Images/val2014/') + imgFilename)

                # Resize images to fit in VGG matrix
                # TODO: not optimal, find ways to pass whole image (padding)
                image_resized = resize(I, (224, 224), anti_aliasing=True)

                image_matrix.append(image_resized)

            # Resizing the shape to have the channels first as keras demands
            image_array = np.rollaxis(np.array(image_matrix), 3, 1)
            return image_array

        elif train_or_val == 'train':

            for index, image_id in tqdm(zip(range(len(img_coco_batch)), img_coco_batch), total=len(img_coco_batch)):
                imgFilename = 'COCO_' + 'train2014' + '_' + str(img_coco_subset[int(index)]).zfill(12) + '.jpg'

                I = io.imread(os.path.join(data_path, 'Images/train2014/') + imgFilename)

                # Resize images to fit in VGG matrix
                # TODO: not optimal, find ways to pass whole image (padding)
                image_resized = resize(I, (224, 224), anti_aliasing=True)

                image_matrix.append(image_resized)

            # Resizing the shape to have the channels first as keras demands
            image_array = np.rollaxis(np.array(image_matrix), 3, 1)
            return image_array

    # Using tensorflow crop

    if tf_pad_resize is True:
        # Start tensorflow session
        sess = K.get_session()

        if train_or_val == 'val':

            image_matrix = []

            for index, image_id in tqdm(zip(range(len(img_coco_batch)), img_coco_batch), total=len(img_coco_batch)):
                imgFilename = 'COCO_' + 'val2014' + '_' + str(img_coco_subset[int(index)]).zfill(12) + '.jpg'

                I = io.imread(os.path.join(data_path, 'Images/val2014/') + imgFilename)

                # Resize images to fit in VGG matrix
                #Get the images as a constant
                tf_images = tf.constant(I)
                image_cropped = tf.image.resize_image_with_crop_or_pad(image=tf_images, target_height=224, target_width=224)

                with sess.as_default():
                    image_array = sess.run(image_cropped)

                image_matrix.append(image_array)

            # Resizing the shape to have the channels first as keras demands
            image_array = np.rollaxis(np.array(image_matrix), 3, 1)

            return image_array

        elif train_or_val == 'train':

            image_matrix = []

            for index, image_id in tqdm(zip(range(len(img_coco_batch)), img_coco_batch), total=len(img_coco_batch)):
                imgFilename = 'COCO_' + 'train2014' + '_' + str(img_coco_subset[int(index)]).zfill(12) + '.jpg'

                I = io.imread(os.path.join(data_path, 'Images/train2014/') + imgFilename)

                # Resize images to fit in VGG matrix
                # Get the images as a constant
                tf_images = tf.constant(I)
                image_cropped = tf.image.resize_image_with_crop_or_pad(image=tf_images, target_height=224,
                                                                       target_width=224)

                with sess.as_default():
                    image_array = sess.run(image_cropped)

                image_matrix.append(image_array)

            # Resizing the shape to have the channels first as keras demands
            image_array = np.rollaxis(np.array(image_matrix), 3, 1)

            return image_array


"""Sums the word vectors of all the tokens in a question
A numpy array of shape: (nb_samples, word_vec_dim)"""

def get_questions_matrix(questions, nlp):
    assert not isinstance(questions, str)
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_matrix = np.zeros((nb_samples, word_vec_dim))
    for i in range(len(questions)):
        tokens = nlp(questions[i])
        for j in range(len(tokens)):
            questions_matrix[i,:] += tokens[j].vector

    return questions_matrix

"""Converts string objects to class labels
	encoder:	a scikit-learn LabelEncoder object
	Output:
	A numpy array of shape (nb_samples, nb_classes)"""

def get_answers_matrix(answers, encoder):
    #assert not isinstance(answers, str)
    y = encoder.transform(answers) #string to numerical class
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    return Y

def get_answers_sum(answers, encoder):
    #assert not isinstance(answers, str)
    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    return Y

"""Returns a time series of word vectors for tokens in the question
	A numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)"""

def get_questions_tensor_timeseries(questions, nlp, timesteps):
    #assert not isinstance(questions, str)
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in range(len(questions)):
        tokens = nlp(questions[i])
        for j in range(len(tokens)):
            if j<timesteps:
                questions_tensor[i,j,:] = tokens[j].vector
    return questions_tensor


