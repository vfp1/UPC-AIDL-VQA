#!/usr/bin/env python

"""
import sys
import os

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
"""
import os

# Adapting to Google Cloud imports
try:
    import src.predict as predict
except:
    import predict

# Preprocess data

try:

    path = "G:/My Drive/Studies/UPC-AIDL/VQA/data/"
    weights = "vgg_weights/VGG_LSTM_20190624-100549-EPOCH_20--BSIZE_50--SUBSET_60000--LOSS_categorical_crossentropy--VGG_w_TRUE--MET_categorical_accuracy--OPT_rmsprop--LR_0.001__TS_0.2___epoch_FINAL.hdf5"
    structure = "vgg_weights/vgg_lstm_structure.json"

    predict.VQA_predict().prediction(data_folder=path, structure=structure, weights=weights)

except:

    path = "/aidl/VQA/data/"
    weights = os.path.join(path, "")

    predict.VQA_predict().prediction(data_folder=path, weights=weights, subset_size=1)