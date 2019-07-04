# UPC-AIDL-VQA

An attempt to address the VQA challenge for the Artificial Intelligence for Deep Learning postgraduate

## GROUP MEMBERS

* Elena Alonso
* David Valls
* Hector Cano
* Victor Pajuelo
* Francesc Guimera

## EXECUTION/TEST LOG

Find the execution/test logs with comments at *report/experiments.xlsx*

## INTRODUCTION

### VQA CHALLENGE- definición

The Visual Question Answering (VQA) Challenge is a task in machine learning where given an image and a natural language question about the image, the trained system provides an accurate natural language answer to the question. The goal is to be able to understand the semantics of scenes well enough to be able to answer open-ended, free-form natural language questions (asked by humans) about images.

The aim of the project developed at AIDL2019 IA with ML team is to design and build a model that is able to perform this task.
To simplify the task we have opted to use the most freqüent answers aproach instead of generating a natural language answer.

The VQA Challenge provides a train, validation and test sets, containing more than 250K images and 1.1M questions. The questions are annotated with 10 concise, open-ended answers each. Annotations on the training and validation sets are publicly available. This datasets will be used to trains and develop the model developed that is the target of this project.

### VQA – ELEMENTS

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

First, we will make a brief description of the environment used to develop, test and run the models.

Two different kinds of environment have been used, on the idea to save the scarce computational resources on Google Cloud as much as possible for the complete dataset model trains and final model tweaking.

The Develop tasks and first train runs (few epochs just to see run ability and debug errors) of each model have been done on a laptop machine with Ubuntu 18.0 and TF/KERAS/PYTHON 3.0. TensorFlow was compiled for the use of the internal GPU NVIDIA provided on this Laptop.

At the beginning, due to GPU’s memory problems the Dataset used was a reduced subset of the original ones only containing a small sample of each of VQA datasets, normally 10.000 images (the size of the batch images can be selected at model by means of a predefined variable). After find the `fit_generator` and  `batch_generator` solution full dataset model has been trained.

Once model is able to run smooth and gives some (usually trashy) results, then the model is passed on Google Cloud environment to take advantage of the available machine and GPU. Once again the approach is to text the model for a few epochs and small dataset, in order to see everything runs well here, and then trained with the full dataset.

This approach has been keep for the dimensional changes of model (number and width of layers, etc.) at the first improvements tests.

With the final model defined, latest tweaks and improvements has been test on Google Cloud Machine.

### LIBRARYS USED

To help on manage data, prepare it, or plot the results, different libraries had been used when developing the models. It is possible to distinguish three types or areas on behalf of the use of the library.

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
    * from keras.models import model_from_json: used to be able to recover net models that are defined in class “models”

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

To introduce the images on our first model based on FCC input layer, We do unroll the images by using the `layer.flatten` Keras tool.

*VGG* - This problem was fixed when after debugging all the process (something that was difficult because it worked with a small dataset well but suddenly give us an error when try was done using more images

### REPRESENTATIVE SUBSET AND BIASED OR WRONG SUBSET SELECTION

Reduction of training sets is necessary due to VQA datasets are huge. The use of all COCO dataset when training models start gives us a GPU memory overflow problem, data occupying all the available memory with the training not able to finish and suddenly come over.

Using fit_generator instruction in Keras to avoid this, was only to discover that the long time required to trains the whole dataset became our next bottleneck and we’re run out of time so we decide to move on small training dataset or subset.

We can define a representative subset of the original dataset COCO, which (should) satisfies three main characteristics:

1. It is smaller in size compared to the original dataset.
2. It captures the most of information from the original dataset compared to any subset of the same size.
3. It has low redundancy among the representatives it contains.

When doing this we should be aware to select a good one subset by taking a number of samples of random, representative and non biased data. We should also removing our data from repetitive behavior by shuffling it, in order to avoid repetitive behaviours of the loss values.

![repetitive behaviours of the loss valuesAlt text](images/loss.png?raw=true "repetitive behaviours of the loss values")

By doing this the subset data should ensure to avoid selection bias. Selection bias occurs when the samples used to produce the model are not fully representative of cases that the model may be used for in the future, particularly with new and unseen data.

After the subset is done we have to split it and use available data to perform the tasks of training, validation and testing where:

* Training set: is a set used for learning and estimating parameters of the model.
* Validation set: is a set used to evaluate the model, usually for model selection.
* Evaluation (testing) set: is a set of examples used to assess the predictive performance of the model.

Finally the used subset is of 25.000 Images (with Ground Truth and text Labels) for Train (80%) and Validation (20%) sets.

Below is shown the distribution of answers for our subset. Notice that _yes_has more than 5000 appearances within the subset.
Therefore, oversampling for the other answers would have been extremely useful.
![LSTM-VGG](images/subset.png?raw=true "LSTM-VGG")
## BASIC MODEL

### MODEL PROPOSAL

Our proposed model (Figure 1) derives inspiration from the winning architecture developed by Teney et al. for the 2017 VQA Challenge. The model implements a joint LSTM/FCC for question and image embeddings, respectively.

It then uses top-down attention, guided by the question embedding, on the image embeddings. The model inputs are pre-processed by means of the aforementioned libraries in order to be able for the model to swallow it.

Model is divided into 3 different parts, each one related to one functional area of the task.

The first one is the part dedicated to the data ingestion that means to collect the training, test and validation datasets from is location, pick the data up from these datasets, and re-ordering and prepare it in order to be suitable for the model to process it.

First difficulty for the team appears here, as the library vqaHelpers referred on models available from previous years, has been done in Python 2, instead of the Python 3 used here.

The image feature vectors along with this question embedding of size number-of-hidden-units (1280) are then pass into FCC layers while Question Inputs are tokenized and represented using word embedding by using the pretrained word2vec from Google and feeded onto an LSTM stack of layers. Then both branches are concatenated and the full ensemble is trained togheter.

### BASIC MODEL

The first attempt to solve VQA and get something usable was a network composed by an MLP FCC for image processing and LSTM’s for NLP, and looks like this:
      Models and script to call them and pass the parameterers are located on.

RESULTS (ACCURACY AND LOSS PLOTS)

## BASIC MODEL ARCHITECTURE VARIATIONS

### MODEL LSTM+VGG

The first variation, as suggested by our team supervisor and logical first step is to change the FCC branch to do the images learning by a more appropriate and natural model of CNN, a VGG – 16 in this case. 

With this change, model looks as :

![LSTM-VGG](images/model_LSTM-VGG.png?raw=true "LSTM-VGG")
![LSTM-VGG](images/model_LSTM-VGGb.png?raw=true "LSTM-VGG")

## MODEL COMPARISON

## ISSUES

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
