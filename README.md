# Multi-Layer Perceptron Neural Network Implementation

This project is a from-scratch implementation of a Multi-Layer Perceptron (MLP) neural network using Python and NumPy. The network is trained with the backpropagation algorithm to predict river flow rates based on historical data.

This work was completed as part of the "AI Methods" (COB107) module at Loughborough University.

## Overview

The goal of this project was to build, train, and evaluate a neural network without relying on high-level machine learning libraries like TensorFlow or PyTorch. The notebook `Backpropagation Neural Network.ipynb` contains the complete implementation, from data preprocessing to model evaluation and visualization.

## Features

* **From-Scratch Implementation:** The neural network's forward and backward passes are built entirely with NumPy.
* **Backpropagation with Momentum:** The model uses the standard backpropagation algorithm with a momentum term to help stabilize and accelerate training.
* **Data Preprocessing:** Implements min-max normalization to scale features appropriately for the sigmoid activation function.
* **Self-Contained Notebook:** The required datasets are loaded directly from this GitHub repository, making the project easily reproducible.
* **Hyperparameter Configuration:** Key parameters like learning rate, number of epochs, and hidden layer size are defined in a dedicated cell for easy experimentation.
* **Performance Evaluation:** The trained model is evaluated on an unseen test set using common regression metrics:
   * Coefficient of Efficiency (CE)
   * R-Squared (R²)
   * Root Mean Squared Error (RMSE)
* **Visualizations:** The training progress (learning curve) and final predictions versus actual values are plotted using Matplotlib for clear analysis.

## How to Run

The easiest way to run this project is by using Google Colab, which provides a free, interactive environment in your browser.

1. **Click the "Open in Colab" badge** at the top of this README.
2. The notebook will open in a new tab. All required data will be loaded automatically from this repository.
3. To experiment, you can change the values in the **"Configuration and Hyperparameters"** cell.
4. To run the entire project, select `Runtime > Run all` from the Colab menu.

The notebook will then execute from top to bottom, displaying the training progress, learning curve, final performance metrics, and a plot of the predicted vs. actual flow rates.

## Project Structure

```
.
├── Backpropagation Neural Network.ipynb  # The main Jupyter Notebook with all the code
├── train_data_lagged.xlsx                # Training dataset
├── validation_data_lagged.xlsx           # Validation dataset
├── test_data_lagged.xlsx                 # Test dataset
└── README.md                             # This file
```

## Coursework Specification

This project fulfills the requirements of the COB107 "ANN Implementation" coursework, which involves:

1. Appropriate data pre-processing.
2. Implementation of the MLP algorithm and its documentation.
3. Training the network and selecting the best model configuration.
4. Evaluating the final model's performance on unseen data.
