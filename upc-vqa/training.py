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
    import src.train as train
except:
    import train

# Preprocess data

try:

    path = "G:/My Drive/Studies/UPC-AIDL/VQA/data/"

    train.VQA_train().train(path, model_type=2, num_epochs=2,
                            subset_size=10, bsize=2, steps_per_epoch=1,
                            keras_loss='categorical_crossentropy',
                            keras_metrics='categorical_accuracy', learning_rate=1e-2,
                            optimizer='rmsprop', fine_tuned=True, test_size=0.20)

except:

    path = "/aidl/VQA/data/"

    train.VQA_train().train(path, model_type=2, num_epochs=200,
                            subset_size=60000, bsize=50, steps_per_epoch=20,
                            keras_loss='categorical_crossentropy',
                            keras_metrics='categorical_accuracy', learning_rate=1e-3,
                            optimizer='rmsprop', fine_tuned=True, test_size=0.20)