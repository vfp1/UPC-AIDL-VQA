# UPC-AIDL-VQA

An attempt to address the **VQA challenge** for the *Artificial Intelligence for Deep Learning* postgraduate.

## GROUP MEMBERS

* Elena Alonso
* David Valls
* Héctor Cano
* Víctor Pajuelo
* Francesc Guimerà

## EXPERIMENTS

Find the description of the tests and experiments, with parameters, graphs and comments at *[experiments.xlsx](report/experiments.xlsx)*

## INTRODUCTION

### VQA CHALLENGE DEFINITION

The Visual Question Answering (VQA) Challenge is a task in machine learning where given an image and a natural language question about the image, the trained system provides an accurate natural language answer to the question. The goal is to be able to understand the semantics of scenes well enough to be able to answer open-ended, free-form natural language questions (asked by humans) about images.

The aim of the project developed at AIDL2019 IA with ML team is to design and build a model that is able to perform this task.
To simplify the task we have opted to use the most frequent answers aproach instead of generating a natural language answer.

The VQA Challenge provides train, validation and test sets, containing more than 250K images and 1.1M questions. The questions are annotated with 10 concise, open-ended answers each. Annotations on the training and validation sets are publicly available. This datasets will be used to trains and develop the model developed that is the target of this project.

### VQA - ELEMENTS

The Visual Question Answering (VQA) problems require the model to understand a text-based question, identify elements of an image, and evaluate how these two inputs relate to one another. Much of the progress in VQA parallels developments made in other problems, such as image captioning and textual question answering.

Several models in deep learning have been developed to combine computer vision components with NLP components in order to solve the VQA Challenge.

These models fit a particular design paradigm, with modules for question encoding, image feature extraction, fusion of the two (that could be powered with attention mechanisms) to finish with a classification task over the space of answer(s).

### IA TOOLS TO APPROACH VQA

The method to approach VQA tasks then is based on three subcomponents:

* Image Representations: creating representations for the image and question.
* A Neural Network, to pass these representations through it, in order to create a co-dependent embedding;
* Natural language answer generation by means of the use of a typical NLP processing path (tokenization and word-embedding)

Finally both parts converge in one by concatenation, and then the result is trained as an ensemble.

## DEVELOP ENVIRONMENT AND LIBS SUMMARY

Two different kinds of environment have been used, on the idea to save the scarce computational resources on Google Cloud as much as possible for the complete dataset model trains and final model tweaking.

The Develop tasks and first train runs (few epochs just to see run ability and debug errors) of each model have been done on a laptop machine with Ubuntu 18.0 and TF/KERAS/PYTHON 3.0. TensorFlow was compiled for the use of the internal GPU NVIDIA provided on this Laptop.

At the beginning, due to GPU's memory problems the Dataset used was a reduced subset of the original ones only containing a small sample of each of VQA datasets (the size of the batch images can be selected at model by means of a predefined variable). After find the `fit_generator` and  `batch_generator` solution full dataset model has been trained.

Once the model is able to run smooth and gives some (usually trashy) results, then the model is uploaded to Google Cloud environment to take advantage of the available machine and GPU.
Once again the approach is to test the model for a few epochs with a small dataset, in order to see everything runs well there, and then trained with the full dataset.

With the final model defined, latest tweaks and improvements has been test on Google Cloud Machine.

### LIBRARYS USED

There is a list and short description of the use of each:

1. DATA INGESTION and DATASET BUILD UP and PREPROCESSING.

    * Datetime: for date and time purposes.
    * Sys, warnings: Interact with OS and manage warning cue.
    * Random: Python Lib. That generated a random seed that will be used later to initialize the weights of the network layers
        * From random import shuffle, sample.
    * vqaHelpers: Custom made lib that has been adapted to run under Python3 with functions implemented to pre-process COCO images and put the relation between an image, his ground truth and corresponding label on a .txt file as a vector so they can be feeded onto the batch generator.
    * Pickle: the pickle lib. implements binary protocols for serializing (flattening) and de-serializing (de-flattening) a Python object structure.
    * GC: Generational garbage collector, to free memory from Python unused objects.
    * Os: The Os module provides a way of using operating system dependent functionalities.
    * numpy: NumPy is the fundamental package for scientific computing with Python and can also be used as an efficient multi-dimensional container of generic data.
    * Pandas: Pandas is an open library providing data structures and data analysis tools for Python programming language.
    * Scipy: The SciPy library provides many numerical routines as for numerical integration, interpolation, optimization, linear algebra and statistics.
    * Src: used as a rood folder for all our function.
    * Utils: This module makes it easy to execute common tasks in Python scripts such as converting text to numbers and making sure a string is in unicode or bytes format.

2. KERAS LIBRARIES (define and fit model )

    * from keras.models import Sequential
    * from keras.models import Model
    * from keras.layers.core import Dense, Dropout, Activation, Reshape
    * from keras.layers.recurrent import LSTM
    * from keras.layers.merge import Concatenate: used to make the union between the image learning branch and the NLP branch. One of the experiments will consist in change the concatenate by a dot product, as suggested by our teacher.
    * progressbar: A Python Progressbar library to provide visual text based progress to long running operations.
    * from sklearn.preprocessing import LabelEncoder: Encode labels with value between 0 and n-1 (when n= number of classes).
    * from keras.utils import plot_model
    * from keras.utils import Sequence
    * from keras.callbacks import TensorBoard: The use of the Keras callbacks to plot the accuracy and loss curves by using Tensorboard application.
    * Models: self-made class with the definition and implementation of the layers of the models trained by the team. The parameters and dimensions of the model are  passed from a script
    * from keras.models import model_from_json: used to be able to recover net models that are defined in class *models*

3. PLOT LIBRARIES

    * SpaCy: spaCy is a Lib. to prepare text for deep learning. It can easily construct statistical models for a variety of NLP problems.
    * Try: Implements the exception handler, in case anything goes wrong.
    * Features: own implemented class with a set of functions that build the tensors to be trained.
    * Graphviz;  This package facilitates the creation and rendering of graph descriptions in the DOT language of the Graphviz graph drawing software from Python.
    * Pydot: Is an interface to Graphviz and can parse and dump into the DOT language used by GraphViz.
    * Sklearn: refere to scikit instead. Used to split the train set.
    * skimage: the Scikit-image library is a collection of algorithms for image processing.

### DEVELOP FIRSTS STEPS AND TROUBLES

Once the draft of the first model is done, both by looking at VQA paper as well as by looking for some available previous models out there, and as soon as we made the firsts attempts to run it, first troubles appear when dealing with dataset to charge it and when running due to inconsistency of the subsets and some libraries dependences and data format incompatibilities.

To introduce the images on our first model based on FCC input layer, we do unroll the images by using the `layer.flatten` Keras tool.

*VGG* This problem was fixed after debugging all the process (something that was difficult because it worked with a small dataset well but suddenly give us an error when we tried done using more images).

### REPRESENTATIVE SUBSET AND BIASED OR WRONG SUBSET SELECTION

Reduction of training sets is necessary due to VQA datasets are huge. The use of all COCO dataset when training models start gives us a GPU memory overflow problem, data occupying all the available memory with the training not able to finish and suddenly come over.

Using `fit_generator` instruction in Keras to avoid this, was only to discover that the long time required to train the whole dataset became our next bottleneck and we've run out of time so we decide to move on small training dataset or subset.

We can define a representative subset of the original dataset COCO, which (should) satisfies three main characteristics:

1. It is smaller in size compared to the original dataset.
2. It captures the most of information from the original dataset compared to any subset of the same size.
3. It has low redundancy among the representatives it contains.

When doing this we should be aware to select a good subset by taking a number of samples of random, representative and non biased data. We should also remove our data from repetitive behavior by shuffling it, in order to avoid repetitive behaviours of the loss values.

By doing this the subset data should ensure to avoid selection bias. Selection bias occurs when the samples used to produce the model are not fully representative of cases that the model may be used for in the future, particularly with new and unseen data.

After the subset is done we have to split it and use available data to perform the tasks of training, validation and testing where:

* **Training set**: is a set used for learning and estimating parameters of the model.
* **Validation set**: is a set used to evaluate the model, usually for model selection.
* **Evaluation (testing) set**: is a set of examples used to assess the predictive performance of the model.

Finally the used subset is of 25.000 Images (with Ground Truth and text Labels) for Train (80%) and Validation (20%) sets.

## NETWORK MODEL

### FIRST MODEL PROPOSAL

Our first proposed model derives inspiration from the winning architecture developed by Teney et al. for the 2017 VQA Challenge. The model implements a joint LSTM/FCC for question and image embeddings, respectively.

It then uses top-down attention, guided by the question embedding, on the image embeddings. The model inputs are pre-processed by means of the aforementioned libraries in order to be able for the model to swallow it.

Model is divided into 3 different parts, each one related to one functional area of the task.

The first one is the part dedicated to the data ingestion that means to collect the training, test and validation datasets from is location, pick the data up from these datasets, and re-ordering and prepare it in order to be suitable for the model to process it.

First difficulty for the team appears here, as the library vqaHelpers referred on models available from previous years, has been done in Python 2, instead of the Python 3 used here.

The image feature vectors along with this question embedding of size number-of-hidden-units (1280) are then pass into MLP FCC layers while Question Inputs are tokenized and represented using word embedding by using the pretrained word2vec from Google and feeded onto an LSTM stack of layers. Then both branches are concatenated and the full ensemble is trained togheter.

## NETWORK MODEL ARCHITECTURE VARIATIONS

### MODEL LSTM+VGG

The first variation, as suggested by our team supervisor and logical first step is to change the FCC branch to do the images learning by a more appropriate and natural model of CNN, a VGG – 16 in this case.

With this change, model looks as :

![LSTM-VGG](images/model_LSTM-VGG.png?raw=true "LSTM-VGG")
![LSTM-VGG](images/model_LSTM-VGGb.png?raw=true "LSTM-VGG")

## DEVELOPMENT

### How did we built the images batch

```python
elif train_or_val == 'train':

    for image_id in tqdm(img_coco_batch, total=len(img_coco_batch)):

        imgFilename = 'COCO_' + 'train2014' + '_' + str(image_id).zfill(12) + '.jpg'

        I = io.imread(os.path.join(data_path, 'Images/train2014/') + imgFilename)

        # Resize images to fit in VGG matrix
        # TODO: not optimal, find ways to pass whole image (padding)
        image_resized = resize(I, (224, 224), anti_aliasing=True)

        if standarization is True:

            # Standarization for zero mean and unit variance
            from matplotlib import pyplot as plt
            scalers = {}

            try:
                # Loop through all the image channels
                for i in range(image_resized.shape[2]):
                    #Do scaling per channel
                    scalers[i] = StandardScaler()
                    image_resized[:, i, :] = scalers[i].fit_transform(image_resized[:, i, :])

                image_matrix.append(image_resized)

            # Some images have a fault in channels (grayscale perhaps)
            # It gets IndexError: tuple index out of range so we pass this
            except IndexError:
                print("Probably a grayscale image:", image_id)

                # Adding channel dimension to a grayscale image
                stacked_img = np.stack((image_resized,)*3, axis=-1)

                print("Shape of reshaped grayscale image:", stacked_img.shape)

                # Loop through all the image channels
                for i in range(stacked_img.shape[2]):
                    # Do scaling per channel
                    scalers[i] = StandardScaler()
                    stacked_img[:, i, :] = scalers[i].fit_transform(stacked_img[:, i, :])

                image_matrix.append(stacked_img)

        elif standarization is False:

            image_matrix.append(image_resized)

    # Resizing the shape to have the channels first as keras demands
    image_array = np.rollaxis(np.array(image_matrix), 3, 1)
    return image_array
```

### Our custom Batch Generator

```python

class Custom_Batch_Generator(Sequence):
    # Here we inherit the Sequence class from keras.utils
    """
    A custom Batch Generator to load the dataset from the HDD in
    batches to memory
    """

    def __init__(self, questions, images, answers, batch_size, lstm_timestep,
                 data_folder, nlp_load, lbl_load, tf_crop, img_standarization):
        """
        Here, we can feed parameters to our generator.
        :param questions: the preprocessed questions
        :param images: the preprocessed images
        :param answers: the preprocessed answers
        :param batch_size: the batch size
        :param lstm_timestep: the timestep of the LSTM timestep = len(nlp(subset_questions[-1]))
        :param data_folder: the data root folder (/aidl/VQA/data/, in Google Cloud
        :param nlp_load: the nlp processing to be loaded from VQA_train
        :param lbl_load: the lbl processing to be loaded from VQA_train
        :param val_coco_id_file: the file for the coco id
        """
        self.questions = questions
        self.images = images
        self.answers = answers
        self.batch_size = batch_size
        self.lstm_timestep = lstm_timestep
        self.data_folder = data_folder
        self.nlp_load = nlp_load
        self.lbl_load = lbl_load
        self.tf_crop = tf_crop
        self.img_standarization = img_standarization

    def __len__(self):
        """
        This function computes the number of batches that this generator is supposed to produce.
        So, we divide the number of total samples by the batch_size and return that value.
        :return:
        """
        print("Batches per epoch:", (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int))

        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):

        """
        Here, given the batch number idx you need to put together a list
        that consists of data batch and the ground-truth (GT).

        In our case, this is a tuple of [questions, images] and [answers]

        In __getitem__(self, idx) function you can decide what happens to your dataset when loaded in batches.
        Here, you can put your preprocessing steps as well.

        :param idx: the batch id
        :return:
        """
        batch_x_questions = self.questions[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_x_images = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_y_answers = self.answers[idx * self.batch_size: (idx + 1) * self.batch_size]

        print("Getting questions batch")
        X_ques_batch_fit = get_questions_tensor_timeseries(batch_x_questions, self.nlp_load, self.lstm_timestep)

        print("Getting images batch")
        X_img_batch_fit = get_images_matrix_VGG(batch_x_images, self.data_folder,
                                                train_or_val='train', standarization=self.img_standarization,
                                                tf_pad_resize=self.tf_crop)

        print("Get answers batch ")
        Y_batch_fit = get_answers_matrix(batch_y_answers, self.lbl_load)

        print(X_ques_batch_fit.shape, X_img_batch_fit.shape, Y_batch_fit.shape)

        return [X_ques_batch_fit, X_img_batch_fit], Y_batch_fit
```

## ISSUES

* **Resize Images**: We had  to resize images form original size to 224x224 to feed them into the input layer of the VGG.
  We found some problems around it due to the fact that some images are portrait oriented and others are landscape.

* **Size of the dataset** (`fit--> fit_generator+batch_generator`): Due to recurrent overflows of memory at GPU we start to look for ways to work with smaller datasets.
  Then we found the solution by means of creating a `batch-generator` and changin the `model.fit` for `model.fit_generator`.

  *This solution was provided by **Mr. Chollet** in a Keras forum at Stackoverflow!!!*

* **Grayscale images**: Due to the fact the we didn't know the dataset very well, We could not foresee that there are some grayscale images.
   These images of only one channel, by entering the VGG (that was expecting a 3 channel RGB image), broke the program and make stop the tranining in progress.

* **Crop from Tensorflow**: We tried to use Tensorflow's Crop instead of the resize, to mantain aspect ratio but we were not able.

  Memory starts to get leaked with the prior epochs data and finally the epochs were much more slower and stopped, completely freezing the computer.
  We have not been able to found were the error with crop lies.

* **Non representative dataset (due to code errors in Python)**: biased answers.

  * When started to make inferences, all the system was able to give us was a very small set of so much space of answers ("1", "Mt. Airy", "Yes").
  * Then after running several trainings without realizing of the problem, but changing parameters trying to make it better and more capable to answer more things we started to be suspicious that our problem was that we were training over a small ann non representative set of samples.
  * The problem was that we were taking the lenght of a list as a value to iterate, but where it was supossed to be a index list, we where taking the length of a string. Due to this we were iterating only over the first positions of the dataset, and the model was learning only a few things.

* **Steps_per_epoch**: Other isue that we have fixed at the end was that the Keras parameter steps_per_epoch was limiting the number of items of the datased percoursed in trainings to 1000 samples. After we realized that epochs were running too fast, we found that the relation between batch_size and steps_epochs was the responsible.
  * The right number for the steps is `np.ceil(subset/batchsize)`.

* **After we fixed both last errors, the prediction has become much better (not the curves!).**

* **Repetibility of tests**: to guarantee the repetibility of tests, we have used `PYTHONHASHSEED` to get the same *random* items in each run.

* **Terminal connection lost**: When a terminal connection is lost any pogram being executed from that terminal is lost. This can be frustrating in programs that take very long to execute, as the trainings. We have used `nohup`and `screen` to avoid this problem.

## NEXT STEPS

* **Evaluation**: We would have liked to have time to run evaluation in our last models, trained with a quite big subset.

* **Subset representativity**: try to make a very representative subset and do same trains in order to see if we can get better inference answers with the same parameters as we did the tests.

* **Underrepresented values**: Try to make oversampling with the less represented values of the dataset.

* **Model Architecture**: Change the model and try some more advanced architectures like *Bi-Lstm*, add *atention* mechanisms and maybe some advance NLP techniques like *transformers*.

* **Portable production model**: We would like to have much more time in order to develop a production model running on some kind of portable platform, and try it out there.

## CONCLUSIONS

Our conclusions are more related about how the time should be dsitributed when planning a machine learning project.

1. Assuming you have datasets available for your purpose, we have spent 2/3 of the available time preparing the data ingestion  and trying to make the model able to deal with the provided prepared data and to run.

    Thinking of more experienced and with better ML skills people doing that we can suposse that at half of the time could be spent doing this, specially if you have to look for your own data.

2. We spent much less time tweaking the model that we thought due mainly to the reason above, and also the time lost on bad trainings (the memory overflows etc..).

    Having better comprension of the behaviour of Keras functions would have save us a lot of processing time.
    In this case, this consuming time cost money, so better save running machine time.

3. It's also difficult to understand the results sometime. To be rigourous when annotating the parameters of each train has been useful when analysing the results and compare them.

4. Also we have get good results by using pretrained model on VGG part. So work more on your dataset and take advantage of the models that had been already trained for you.

5. The fun part of the project was to play with the model when was running, but hard work has to be done in order to get into this.

> **The main conclusion could be...take care of your datasets!!!**

## HOW TO RUN THIS REPO

### English model from Spacy

```python
python -m spacy download en
```

### VQA helpers

This installs the helpers for _ingestion_ and _evaluation_ of the VQA Challenge. This is based on the
[this repository](https://github.com/vfp1/VQA).

``` python
python -m pip install installers/helpers/dist/vqaHelpers-0.4_faster_unzip_capabilities-py3-none-any.whl
```

### Requirements

``` python
pip install -r requirements.txt
```

### Preparing the datasets

The datasets are used as in the [VQA Challenge](https://visualqa.org/download.html).

The datasets must be downloaded and unzipped.

Edit `upc-vqa/data_preparation.py` to update the `path_data` to point to your data path.
That script can be also used to download and unzip the data.
Beware, it takes time and a lot of disk space.
It is **30GB** after all.

Run `upc-vqa/data_preparation.py` to prepare the data.

``` python
python upc-vqa/data_preparation.py
```

### Training

Edit `upc-vqa/training.py` to define your training parameters.

Run `upc-vqa/training.py` to train.

You need to set the `PYTHONHASHSEED` variable to 42, so that all the subsets from VQA and the train/validation splits are consistent across the experiments.

``` python
\$ PYTHONHASHSEED=42 python upc-vqa/training.py
```

### Experiment setting

Each training run will output a folder with the date of run and an unique identifier. Whithin each folder you will find:

* Training/Validation Loss & Accuracy charts
* Tensorboard logs
* Model parameters.json to be loaded after in predict/evaluation
* Model weights.hdf5 to be loaded after in predict/evaluation
* A CSV with all the parameters of the training being:
  * param unique_id: the unique id for the experiment
  * param data_folder: the root data folder
  * param model_type: 1, MLP, LSTM; 2, VGG, LSTM
  * param num_epochs: the number of epochs
  * param subset_size: the subset size of VQA dataset, recommended 25000 by default
  * param subset: whether to subset the dataset or not
  * param bsize: the batch size, default at 256
  * prarm auto_steps_per_epoch: automated steps per epoch counter to adjust number samples/number batch
  * param steps_per_epoch: the steps for each epoch
  * param keras_loss: the chosen keras loss
  * param keras_metrics: the chosen keras metrics
  * param learning_rate: the chosen learning rate
  * param optimizer: the chosen optimizer
  * param fine_tuned: whether to fine tune VGG or not
  * param test_size: the split test size, set to 80/20
  * param vgg_frozen: the number of frozen layers
  * param lstm_hidden_nodes: the LSTM hidden nodes, set to 512
  * param lstm_num_layers: the number of chosen LSTM layers
  * param fc_hidden_nodes: the number of FC hidden nodes after model merges, set to 1024
  * param fc_num_layers: the number of FC layers (DENSE)
  * param merge_method: the chosen merge method, either concatenate or dot
  * param tf_crop_bool: True/False cropping the images with tensorflow (True) or scikit image (False)
  * param image_standarization: whether to do image scaling for zero mean and unit variance
  * param vgg_finetuned_dropout: the dropout for the fine tuned VGG
  * param vgg_finetuned_activation: the activation for the fine tuned VGG
  * param merged_dropout_num: the dropout for the merged part
  * param merged_activation: the activation function for the merged part
  * param finetuned_batchnorm: the batchnorm for the fine tuned part
  * param merged_batchnorm: the batchnorm for the merged part

### Predicting

Edit `upc-vqa/predicting.py` to assign how many images you want to predict on. You need to specify the folder with the results.

Run `upc-vqa/predicting.py` to train.

``` python
python upc-vqa/predicting.py
```
