# Spiral Classification with TensorFlow

A simple neural network that classifies spiral data using TensorFlow and Keras.

## Overview
This project trains a feed-forward neural network on a spiral dataset.  
The model learns to separate two intertwined spirals using engineered features and sigmoid activations.

**Features used**
- X₁  
- X₂  
- X₁²  
- X₂²  
- sin(X₁)  
- sin(X₂)

**Model configuration**
- One hidden layer with 6 neurons  
- Sigmoid activation for all layers  
- SGD optimizer (learning rate = 0.1)  
- Binary cross-entropy loss  
- Early stopping based on validation loss

## Requirements
* TensorFlow (includes Keras)
* NumPy
* Pandas
* scikit-learn
* Matplotlib (for visualization)
