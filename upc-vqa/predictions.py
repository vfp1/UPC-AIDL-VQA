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
    weights = "output/VGG_LSTM/reports/20190630-083118-_666b9c1b/VGG_LSTM_WEIGHTS_uuid_666b9c1b.hdf5"
    structure = "output/VGG_LSTM/reports/20190630-083118-_666b9c1b/vgg_lstm_structure_666b9c1b.json"

    predict.VQA_predict().prediction(data_folder=path, structure=structure, weights=weights)

except:

    path = "/aidl/VQA/data/"
    weights = os.path.join(path, "")

    predict.VQA_predict().prediction(data_folder=path, structure=structure, weights=weights, subset_size=1)