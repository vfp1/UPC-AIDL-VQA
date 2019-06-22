#!/usr/bin/env python

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import vqa_data_prep

__author__ = "Victor Pajuelo Madrigal"
__copyright__ = "Copyright 2019, UPC Group"
__credits__ = ["Victor Pajuelo Madrigal", "Jiasen Lu", "@abhshkdz"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Victor Pajuelo Madrigal"
__email__ = "-"
__status__ = "Development"

# Preprocess data

path_data = "G:/My Drive/Studies/UPC-AIDL/VQA/v2/"

path_data_2 = "G:/My Drive/Studies/UPC-AIDL/VQA/data/"

"""
vqa_data_prep.VQA_Preparation().download_vqa(data_folder=path_data,
                                             download_annotations=False, force_reextract=True,
                                             download_COCO_images=True)
"""

vqa_data_prep.VQA_Preprocessing().vqa_text_preprocess(data_folder=path_data_2)