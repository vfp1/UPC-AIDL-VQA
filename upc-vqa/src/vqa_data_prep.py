#!/usr/bin/env python

__author__ = "Victor Pajuelo Madrigal"
__copyright__ = "Copyright 2019, UPC Group"
__credits__ = ["Victor Pajuelo Madrigal", "Jiasen Lu", "@abhshkdz"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Victor Pajuelo Madrigal"
__email__ = "-"
__status__ = "Development"

# Download the VQA Questions from http://www.visualqa.org/download.html
import os
import wget
import zipfile
from tqdm import tqdm

import json   #parse json
import spacy  #tokenizing text
from utils import most_freq_answer

class DirectoryTools(object):
    """
    This class deals with all the folder creation tools
    """

    def folder_creator(self, input_folder):
        """
        This function creates a folder with a given input path

        :param input_folder: the full path of the folder to be created
        :return: the path to the input folder and the folder created
        """
        try:
            if not os.path.exists(input_folder):
                os.makedirs(input_folder)
        except OSError:
            print('Error: DirectoryTools.folder_creator failed to create:' + input_folder)

        return input_folder

class VQA_Preparation(object):
    """
    Download the VQA Questions from http://www.visualqa.org/download.html
    """

    def download_vqa(self, data_folder, download_annotations=True,
                     force_reextract=False, download_COCO_images=False):

        """
        Downloads VQA 2.0 Questions, Annotations and Images
        :param data_folder:
        :param download_annotations:
        :param force_reextract:
        :param download_COCO_images:
        :return: unzipped VQA 2.0 Questions and Annotations
        """

        dir_tools = DirectoryTools()
        self.zip_folder = dir_tools.folder_creator(input_folder=os.path.join(data_folder, 'zips'))
        self.annotations_folder = dir_tools.folder_creator(input_folder=os.path.join(data_folder, 'annotations'))
        self.images_folder = dir_tools.folder_creator(input_folder=os.path.join(data_folder, 'images'))

        self.zip_questions = dir_tools.folder_creator(input_folder=os.path.join(self.zip_folder, 'questions'))
        self.zip_annotations = dir_tools.folder_creator(input_folder=os.path.join(self.zip_folder, 'annotations'))
        self.zip_images = dir_tools.folder_creator(input_folder=os.path.join(self.zip_folder, 'images'))


        question_files = ["v2_Questions_Train_mscoco.zip", "v2_Questions_Val_mscoco.zip", "v2_Questions_Test_mscoco.zip"]

        annotation_files = ["v2_Annotations_Train_mscoco.zip", "v2_Annotations_Val_mscoco.zip"]

        image_files = ["train2014.zip", "val2014.zip", "test2015.zip"]

        if download_annotations is True:
            # Downloading question files
            for file in question_files:

                if os.path.exists(os.path.join(self.zip_questions, file)):

                    print("Path exists")

                    if force_reextract is True:

                        print("Force reextract is True, extracting again")

                        # Unzip the questions
                        for file in os.listdir(self.zip_questions):

                            if file.endswith(".zip"):
                                print("Exctracting {}, please wait with a cup of coffe".format(file))

                                zip_ref = zipfile.ZipFile(os.path.join(self.zip_questions, file), 'r')

                                zip_ref.extractall(self.annotations_folder)

                                zip_ref.close()

                    elif force_reextract is False:
                        pass

                else:

                    # Download the VQA v2.0 Questions
                    wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
                                  out=self.zip_questions)
                    wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
                                  out=self.zip_questions)
                    wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
                                  out=self.zip_questions)

                    # Unzip the questions
                    for file in os.listdir(self.zip_questions):

                        if file.endswith(".zip"):
                            print("Exctracting {}, please wait with a cup of coffe".format(file))

                            zip_ref = zipfile.ZipFile(os.path.join(self.zip_questions, file), 'r')

                            zip_ref.extractall(self.annotations_folder)

                            zip_ref.close()

            # Downloading annotation files
            for file in annotation_files:

                if os.path.exists(os.path.join(self.zip_annotations, file)):

                    print("Path exists")

                    if force_reextract is True:

                        print("Force reextract is True, extracting again")

                        # Unzip the annotations
                        for file in os.listdir(self.zip_annotations):

                            if file.endswith(".zip"):
                                print("Exctracting {}, please wait with a cup of coffe".format(file))

                                zip_ref = zipfile.ZipFile(os.path.join(self.zip_annotations, file), 'r')

                                zip_ref.extractall(self.annotations_folder)

                                zip_ref.close()

                    elif force_reextract is False:
                        pass

                else:

                    # Download the VQA v2.0 Annotations
                    wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
                                  out=self.zip_annotations)
                    wget.download("https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
                                  out=self.zip_annotations)

                    # Unzip the annotations
                    for file in os.listdir(self.zip_annotations):

                        if file.endswith(".zip"):
                            print("Exctracting {}, please wait with a cup of coffe".format(file))

                            zip_ref = zipfile.ZipFile(os.path.join(self.zip_annotations, file), 'r')

                            zip_ref.extractall(self.annotations_folder)

                            zip_ref.close()

        elif download_annotations is False:

            print("Annotations were not downloaded")
            pass

        if download_COCO_images is True:

            # Downloading annotation files
            for file in image_files:

                if os.path.exists(os.path.join(self.zip_images, file)):

                    print("Path exists")

                    if force_reextract is True:

                        print("Force reextract is True, extracting again")

                        # Unzip the annotations
                        for file in os.listdir(self.zip_images):

                            if file.endswith(".zip"):
                                print("Exctracting {}, please wait with a cup of coffe".format(file))

                                zip_ref = zipfile.ZipFile(os.path.join(self.zip_images, file), 'r')

                                zip_ref.extractall(self.images_folder)

                                zip_ref.close()

                    elif force_reextract is False:
                        pass

                else:

                    # Download the VQA v2.0 Annotations
                    print("Downloading train2014.zip")
                    wget.download("http://images.cocodataset.org/zips/train2014.zip", out=self.zip_images)
                    print("Downloading val2014.zip")
                    wget.download("http://images.cocodataset.org/zips/val2014.zip", out=self.zip_images)
                    print("Downloading test2015.zip")
                    wget.download("http://images.cocodataset.org/zips/test2015.zip", out=self.zip_images)

                    # Unzip the annotations
                    for file in os.listdir(self.zip_images):

                        if file.endswith(".zip"):
                            print("Exctracting {}, please wait with a cup of coffe".format(file))

                            zip_ref = zipfile.ZipFile(os.path.join(self.zip_images, file), 'r')

                            zip_ref.extractall(self.images_folder)

                            zip_ref.close()

class VQA_Preprocessing(object):
    """
    Preprocess VQA dataset
    """

    def vqa_text_preprocess(self, data_folder):
        """
        Text processing for VQA

        :param data_folder: the root folder with the downloaded data
        :return:
        """

        try:
            nlp = spacy.load("en_core_web_md")
        except:
            nlp = spacy.load("en_core_web_sm")

        dir_tools = DirectoryTools()
        self.preprocessed = dir_tools.folder_creator(input_folder=os.path.join(data_folder, 'preprocessed'))

        # Create all relevant data-dumps required
        image_set_id = open(os.path.join(self.preprocessed, 'val_images_coco_id.txt'), 'wb')
        ann = os.path.join(data_folder, '_annotations/v2_mscoco_val2014_annotations.json')
        ques = os.path.join(data_folder, '_annotations/v2_OpenEnded_mscoco_val2014_questions.json')
        ques_compile = open(os.path.join(self.preprocessed, 'ques_val.txt'), 'wb')
        ques_id = open(os.path.join(self.preprocessed, 'ques_val_id.txt'), 'wb')
        ques_len = open(os.path.join(self.preprocessed, 'ques_val_len.txt'), 'wb')
        answer_train = open(os.path.join(self.preprocessed, 'answer_val.txt'), 'wb')

        ques = json.load(open(ques, 'r'))
        questions = ques['questions']
        qa = json.load(open(ann, 'r'))
        annotations = qa['annotations']

        print("Begin Data Dump...")

        for index, question in tqdm(zip(range(len(questions)), questions), total=len(questions)):

            ques_compile.write((question['question'] + '\n').encode('utf8'))
            ques_len.write((str(len(nlp(question['question']))) + '\n').encode('utf8'))
            ques_id.write((str(question['question_id']) + '\n').encode('utf8'))
            image_set_id.write((str(question['image_id']) + '\n').encode('utf8'))
            answer_train.write(most_freq_answer(annotations[index]['answers']).encode('utf8'))
            answer_train.write('\n'.encode('utf8'))

        print("Data dump can be found in ../preprocessed/")