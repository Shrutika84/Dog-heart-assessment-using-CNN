# Dog Cardiomegaly Classification using CNN

This project implements a custom Convolutional Neural Network (CNN) to classify dog cardiomegaly images into four categories: **Normal**, **Small Cardiomegaly**, **Large Cardiomegaly**, and **Severe Cardiomegaly**. The model achieves an accuracy of over **76.7%** on the test dataset.

Using **PyTorch** and **Torchvision**, this deep learning model is trained on a dataset of canine heart X-ray images, leveraging techniques such as **data augmentation**, **residual connections**, and **squeeze-and-excitation (SE) blocks** for improved feature recalibration. The project also includes **adversarial testing** to assess the model's robustness and **Grad-CAM** visualizations for model interpretability, ensuring transparency in its decision-making process.

This repository provides all necessary code, instructions for data preprocessing, training, testing, and practical deployment considerations for integrating the model into clinical veterinary environments.


## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Results](#results)

## Introduction

Cardiomegaly, the enlargement of the heart, is a life-threatening condition that can lead to serious health complications in dogs. Early detection of cardiomegaly can prevent such complications. This project focuses on classifying dog heart images based on the degree of cardiomegaly using deep learning models.

The goal of this project is to build a custom CNN model that classifies images into different categories based on the severity of cardiomegaly.

## Dataset

The dataset contains images of dog hearts labeled into the following categories:
Train : 
  - **Normal**
  - **Small Cardiomegaly**
  - **Large Cardiomegaly**

Test :
  - Images


## Model Architecture

The model is a custom Convolutional Neural Network (CNN) built using **PyTorch** with the following features:
- **Residual Blocks**: To facilitate deeper network training by adding skip connections.
- **Squeeze-and-Excitation (SE) Blocks**: For enhanced feature recalibration.
- **Data Augmentation**: Random transformations applied to improve generalization.

### Model Flow:
1. **Input**: RGB image (224x224px).
2. **Feature Extraction**: A series of convolutional and residual blocks.
3. **Feature Pooling**: Global average pooling to reduce the output size.
4. **Classification**: Fully connected layers that lead to the final output classes.

### Residual Block Example:
The residual block contains convolutional layers with batch normalization and activation (ReLU). Skip connections are added to enable better gradient flow and help train deeper networks.


## Usage

1. [Download the labeled training, validation, and unlabeled test dataset here](https://yuad-my.sharepoint.com/personal/youshan_zhang_yu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyoushan%5Fzhang%5Fyu%5Fedu%2FDocuments%2FDog%5FX%5Fray&ga=1)
2. Train the model with the labeled training dataset and validation dataset

## Evaluation
1. **Training Evaluation**
During training, the model is evaluated on the validation set after each epoch. The following metrics are logged:

Training Accuracy: Accuracy on the training set.
Validation Accuracy: Accuracy on the validation set, which is used for early stopping.
Loss: The loss value for both the training and validation sets.

2. **Test Evaluation**
After training is complete, the model is evaluated on a separate test set to measure its generalization. The predicted class labels for each test image are saved in the test_predictions.csv file.

## Model Weights

The trained model weights are available for download from Google Drive. You can download the model weights (`best_model.pth`) using the link below:

- [Download the trained model weights : (Custom_CNN_model.pth)](https://drive.google.com/file/d/1gndENAaFLNXBwLSBZGC7Br4GOsLzaLi4/view?usp=sharing)


## Results
The model achieved an accuracy of 78% on the validation dataset after training for 150 epochs. Below are the key results obtained from the model's evaluation:

**Model Performance:**
Training Accuracy: 77%
Validation Accuracy: 78%
Test Set Accuracy: Available in custom_cnn.csv.

**Performance Metrics:**
To evaluate the model's classification performance, we used the following metrics:

**Accuracy**: Percentage of correctly classified images.
**Precision**: Measures the model's ability to correctly identify positive samples.
**Recall**: Measures the model's ability to identify all relevant instances.
**F1-Score**: A balanced measure of precision and recall.



## Related Publication

For a detailed explanation of the model and its applications, refer to my published paper:

[Convolutional Neural Network for Dog Heart Disease Assessment](https://www.researchgate.net/publication/385896627_Convolutional_Neural_Network_for_Dog_Heart_Disease_Assessment)

This paper discusses the architecture, performance, and practical deployment of the CNN model for classifying canine heart disease conditions, achieving a classification accuracy of 76.7\% on the test set.
