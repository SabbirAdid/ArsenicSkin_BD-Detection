# ArsenicSkin_BD-Detection
Image Classification and Analysis of Arsenicosis affected Skin of Bangladeshi People using Deep Learning

# Image Classification using ResNet-18 & MobileNetV4

This project contains two models, **ResNet-18** and **MobileNetV4**, implemented for image classification tasks using PyTorch. The models are trained on custom datasets, and the dataset consists of images that are classified into two categories: "notinfected" and "infected".

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
3. [Running the Notebooks](#running-the-notebooks)
4. [Dependencies](#dependencies)
5. [Evaluation](#evaluation)

## Overview

### ResNet-18
ResNet-18 is a deep convolutional neural network that has been pre-trained on the ImageNet dataset. In this notebook, we fine-tune ResNet-18 on a custom image dataset. The custom dataset includes images of infected and non-infected categories, and the model is trained to classify these images accurately.

### MobileNetV4
MobileNetV4 is another lightweight convolutional neural network model designed for mobile and edge devices. Similar to ResNet-18, MobileNetV4 is fine-tuned for classifying images from the same custom dataset.

Both models follow a similar process, where we load the dataset, perform image transformations, define a custom dataset class, and then fine-tune the models.

## Setup

### Prerequisites

Ensure that the following libraries are installed:
- PyTorch
- pandas
- PIL (Python Imaging Library)
- torchvision
- numpy

You can install the necessary libraries using `pip`:

```bash
pip install torch torchvision pandas Pillow numpy
```

## Running the Notebooks

### ResNet-18
1. Open the `ResNet18.ipynb` notebook.
2. Load the dataset and apply the necessary transformations.
3. Train the model on the custom dataset.
4. Evaluate the model performance on the test set.

### MobileNetV4
1. Open the `MobileNetv4.ipynb` notebook.
2. Similar to ResNet-18, load the dataset and apply transformations.
3. Fine-tune the MobileNetV4 model on the custom dataset.
4. Evaluate the model's performance.

## Dependencies

The following Python packages are required for this project:
- `torch`
- `torchvision`
- `pandas`
- `Pillow`
- `numpy`

## Evaluation

Both models are trained to classify images into the following categories:
- **notinfected** (label 0)
- **infected** (label 1)

Upon running the notebooks, the models will output the number of samples in the training and testing datasets, and evaluation metrics such as accuracy or loss.
