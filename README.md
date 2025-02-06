# Tumor Classification using Transformer based Neural Network

## Introduction

During my masters degree, I had the opportunity to learn the theory behind a number of very interesting inference models. Unfortunately, I didn't get the opportunity to build any of these interesting models. During free time I have been looking towards building some of these models using some interesting datasets.

## The Dataset and Problem

The dataset and the problem are relatively popular within the image classification space. The dataset used is a selection of MRI images of brains with and without tumors. The goal of the model is to detect whether or not the image is of a "Healthy" brain or if the image features a "Tumor". The dataset features 5000 files and the were 2000 are healthy and the remaining 3000 feature tumors.

The dataset isn't included in the Github repository due to size limitations but can be found here: https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri

## Image Processing

The first step to training this model is to run the 'create_dataset.py' file. This takes a given folder with the name 'mri-tumor' with a number of subfolders outlining the respective classes, and splits them into new subfolders called 'train', 'test', and 'val' in the output folder 'mri-tumor-org. These three folders represent 70%, 20%, and 10% of the data respectively, and are used to either train the model, validate the results from the trained model, or test the results from the trained model. This file also does some other things like pruning through the folders to ensure there are no strange hidden files.

## Training the Model

### Network Architecture

Training the model requires two files, network_architecture.py and train.py, the former illustrates the transform pipeline applied to the images which includes colour jitter, random horizontal flipping, and random vertical flipping, to pad out the training dataset. It also imports the model used for transfer learning, defines the architecture for the head of this network, and sets some constants such as the learning rate and the optimizer. 

### The Train Function

The train.py file outlines the train function which takes the model, optimizer, loss function, train dataset loader and validation dataset loader all defined in the network architecture. Alongside this, it also defines the number of epochs the model will train on, whether or not the model is trained on GPU or CPU, and finally where the model is to be saved. This train function, works by optimizing the imported model for the specific use case, and returns the necessary descriptive statistics. This includes the training loss, validation loss, and the accuracy on the validation set. 

## Testing the Model and Results

### Testing the Model

### Results

## Making Predictions with the Model

I imagine that in a clinical setting, this model could exist on a computer that reads in an MRI image and aids a doctor to understand what is featured. To emulate this, I created predict.py, which takes a given image and responds with whether or not the model thinks the image features a brain tumor or not.