# Logistic Regression From Scratch

A binary logistic regression model implemented from scratch using Python and NumPy.

This project was built to understand the mathematical machinery behind logistic regression rather than relying on a machine learning library such as scikit-learn.

The model predicts penguin gender using physical measurements from the Palmer Penguins dataset.

## Features

The model uses:

* Bill length
* Bill depth
* Flipper length

The target variable is penguin gender:

* `1` = male
* `0` = female

## How It Works

The implementation follows the logistic regression pipeline:

1. Load and clean the dataset.

2. Standardize the input features.

3. Compute the linear combination:

   `z = Xw + b`

4. Apply the sigmoid function:

   `p = 1 / (1 + e^(-z))`

5. Calculate binary cross-entropy loss.

6. Calculate the gradients for the weights and bias.

7. Update the parameters using gradient descent.

8. Convert predicted probabilities into binary classifications using a threshold of `0.5`.

9. Calculate training accuracy.

## Feature Standardization

Each feature is standardized using:

`x_standardized = (x - mean) / standard_deviation`

## Gradient Descent

The model updates its parameters using:

`w = w - learning_rate * dw`

`b = b - learning_rate * db`

The gradients used are:

`dw = (1 / n) * X.T @ (p - y)`

`db = mean(p - y)`

## Loss Function

**Binary cross-entropy** is used as the loss function:

`-mean(y * log(p) + (1 - y) * log(1 - p))`

The predicted probabilities are clipped slightly away from `0` and `1` to prevent numerical issues when calculating logarithms.

For example: 

`log(0)`

## Technologies

* Python
* NumPy
* pandas

## Dataset

This project uses the Palmer Penguins dataset.

The relevant columns are:

* `bill_length_mm`
* `bill_depth_mm`
* `flipper_length_mm`
* `sex`

## Current Limitations

This first implementation is intentionally simple and focuses on understanding the core algorithm.

Currently:

* Accuracy is measured on the training data.
* There is no train/test split.
* There is no automatic convergence.
* The model is not implemented as a reusable class.

These are potential improvements for later versions.

## Purpose

The purpose of this project is not to create the most optimized logistic regression implementation.

It is to understand how logistic regression works internally, including the relationship between linear algebra, probability, binary cross-entropy, gradients, and gradient descent.
