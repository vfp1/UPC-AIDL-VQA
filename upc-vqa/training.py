#!/usr/bin/env python

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import train

# Preprocess data

path = "G:/My Drive/Studies/UPC-AIDL/VQA/data/"

train.VQA_train().train(path)