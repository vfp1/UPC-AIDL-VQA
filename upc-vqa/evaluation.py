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
    import src.evaluate as evaluate
except:
    import evaluate

# Preprocess data

try:

    path = "G:/My Drive/Studies/UPC-AIDL/VQA/data/"
    time_and_uuid = "20190702-223650-_a035d11a"

    evaluate.VQA_evaluate().evaluate(data_folder=path, time_and_uuid=time_and_uuid, subset_size=10, model_type=2)

except:

    path = "/aidl/VQA/data/"
    time_and_uuid = "20190701-142414-_808fd7e5"

    evaluate.VQA_evaluate().evaluate(data_folder=path, time_and_uuid=time_and_uuid, model_type=2)
