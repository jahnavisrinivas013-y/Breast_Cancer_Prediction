# Breast Cancer Prediction Using Machine Learning and Deep Learning

## Project Overview

This project predicts whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** using the Breast Cancer Wisconsin dataset available in Scikit-learn.

The project initially used a traditional Machine Learning approach with **Logistic Regression**. It has now been enhanced by adding a **Deep Learning Artificial Neural Network (ANN)** model.

Both models are evaluated and compared using multiple performance metrics.

---

## Objectives

- Predict whether a breast tumor is malignant or benign.
- Apply data preprocessing and feature scaling.
- Implement a traditional Machine Learning model.
- Implement a Deep Learning Artificial Neural Network.
- Compare the performance of ML and DL models.
- Evaluate the models using accuracy, precision, recall, F1-score and ROC-AUC.
- Visualize model performance using graphs and a confusion matrix.

---

## Dataset

The project uses the **Breast Cancer Wisconsin dataset** provided by Scikit-learn.

### Dataset Details

- Number of samples: **569**
- Number of features: **30**
- Classification type: **Binary Classification**
- Classes:
  - Malignant
  - Benign

The features describe characteristics of cell nuclei obtained from breast tissue measurements.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib
- Seaborn

---

## Machine Learning Model

The original project uses:

### Logistic Regression

Logistic Regression is used as the baseline Machine Learning model for binary classification.

The model achieved:

**Accuracy: 98.25%**

---

## Deep Learning Model

The upgraded project uses an **Artificial Neural Network (ANN)** implemented using TensorFlow and Keras.

### ANN Architecture

```text
Input Layer
    ↓
Dense Layer - 64 neurons
    ↓
ReLU Activation
    ↓
Dropout - 30%
    ↓
Dense Layer - 32 neurons
    ↓
ReLU Activation
    ↓
Dropout - 20%
    ↓
Dense Layer - 16 neurons
    ↓
ReLU Activation
    ↓
Output Layer - 1 neuron
    ↓
Sigmoid Activation
    ↓
Malignant / Benign