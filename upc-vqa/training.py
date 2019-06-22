#!/usr/bin/env python

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import train

__author__ = "Victor Pajuelo Madrigal"
__copyright__ = "Copyright 2019, UPC Group"
__credits__ = ["Victor Pajuelo Madrigal", "Jiasen Lu", "@abhshkdz"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Victor Pajuelo Madrigal"
__email__ = "-"
__status__ = "Development"

# Preprocess data

path = "G:/My Drive/Studies/UPC-AIDL/VQA/data/"

train.VQA_train().train(path)