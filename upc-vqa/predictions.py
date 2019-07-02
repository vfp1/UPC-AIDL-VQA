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
    time_and_uuid = "20190701-142414-_808fd7e5"

    predict.VQA_predict().prediction(data_folder=path, time_and_uuid=time_and_uuid, subset_size=5, model_type=2)

except:

    path = "/aidl/VQA/data/"
    time_and_uuid = "20190701-142414-_808fd7e5"

    predict.VQA_predict().prediction(data_folder=path, time_and_uuid=time_and_uuid, model_type=2)
