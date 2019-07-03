#!/usr/bin/env python

"""
import sys
import os

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
"""

# Adapting to Google Cloud imports
try:
    import src.data_explore as data_explore
except:
    import data_explore

# Preprocess data

import uuid

unique_id = str(uuid.uuid4())[:8]

try:

    path = "G:/My Drive/Studies/UPC-AIDL/VQA/data/"

    data_explore.VQA_explore().explore(unique_id=unique_id, data_folder=path, model_type=2, num_epochs=2,
                            subset_size=25000, subset=True, bsize=2, auto_steps_per_epoch=True, steps_per_epoch=10,
                            keras_loss='categorical_crossentropy',
                            keras_metrics='categorical_accuracy', learning_rate=1e-2,
                            optimizer='rmsprop', fine_tuned=True, test_size=0.20, vgg_frozen=4,
                            lstm_hidden_nodes=512, lstm_num_layers=3, fc_hidden_nodes=1024, fc_num_layers=3,
                            merge_method='concatenate', tf_crop_bool=False, image_standarization=True,
                            vgg_finetuned_dropout=0.5, vgg_finetuned_activation='relu',
                            merged_dropout_num=0.5, merged_activation='tanh',
                            finetuned_batchnorm=True, merged_batchnorm=True)

except:

    path = "/aidl/VQA/data/"

    data_explore.VQA_explore().explore(unique_id=unique_id, data_folder=path, model_type=2, num_epochs=25,
                            subset_size=25000, subset=True, bsize=10, auto_steps_per_epoch=True,steps_per_epoch=20,
                            keras_loss='categorical_crossentropy',
                            keras_metrics='categorical_accuracy', learning_rate=1e-4,
                            optimizer='rmsprop', fine_tuned=True, test_size=0.20, vgg_frozen=4,
                            lstm_hidden_nodes=512, lstm_num_layers=3, fc_hidden_nodes=1024, fc_num_layers=3,
                            merge_method='concatenate', tf_crop_bool=False, image_standarization=True,
                            vgg_finetuned_dropout=0.5, vgg_finetuned_activation='relu',
                            merged_dropout_num=0.5, merged_activation='tanh',
                            finetuned_batchnorm=True, merged_batchnorm=True)