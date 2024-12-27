import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from utils import *

# %% Mean Squared Error GD
# Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform gradient descent to minimize the mean squared error.

        Parameters:
            y (numpy array): The target values.
            tx (numpy array): The input data.
            initial_w (numpy array): The initial weights.
            max_iters (int): The maximum number of iterations.
            gamma (float): The learning rate.

        Returns:
            numpy array: The final weights after gradient descent.
            numpy array: The log of mean squared error at each iteration.
    """
    w = initial_w
    error = y - tx @ w
    loss = 0.5 * np.mean(error**2)
    for _ in range(max_iters):
        grad = -tx.T @ (y - tx @ w) / len(y)
        w = w - gamma * grad
        loss = (y - tx @ w).T @ (y - tx @ w) / (2 * len(y))
    return w, loss


# %% Mean Squared Error SGD
# Linear regression using stochastic gradient descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Perform Stochastic Gradient Descent (SGD) to minimize the Mean Squared Error (MSE).

    Parameters:
        y (numpy.ndarray): The target values.
        tx (numpy.ndarray): The input data.
        initial_w (numpy.ndarray): The initial weights.
        max_iters (int): The maximum number of iterations.
        gamma (float): The learning rate.

    Returns:
        numpy.ndarray: The final weights after SGD.
        numpy.ndarray: The log of MSE errors for each iteration.
    """
    w = initial_w
    for _ in range(max_iters):
        k = np.random.randint(len(y))
        error = y[k] - tx[k, :] @ w
        grad = -tx[k, :].T * error
        w = w - gamma * grad
        loss = 0.5 * np.mean((y[k] - tx[k, :] @ w) ** 2)
    return w, loss


# %% Least Squares
# Least squares regression using normal equations
def least_squares(y, tx):
    """
    Calculate the least squares solution to a linear matrix equation.

    Parameters:
        y (numpy.ndarray): The output vector of shape (n_samples,).
        tx (numpy.ndarray): The input matrix of shape (n_samples, n_features).

    Returns:
        tuple: A tuple containing:
            - beta (numpy.ndarray): The optimal weights of shape (n_features,).
            - loss (float): The mean squared error loss.
    """

    beta = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    loss = 0.5 * np.mean((y - tx @ beta) ** 2)
    return beta, loss


# %% Ridge Regression
# Ridge regression using normal equations
def ridge_regression(y, tx, lambda_):
    """
    Perform ridge regression to compute the weights (beta) and the loss.

    Parameters:
        y (numpy.ndarray): The target values.
        tx (numpy.ndarray): The input data matrix.
        lambda_ (float): The regularization parameter.

    Returns:
        tuple: A tuple containing the weights (beta) and the loss.

    """
    beta = (
        np.linalg.inv(tx.T @ tx + 2 * lambda_ * np.eye(tx.shape[1]) * len(y)) @ tx.T @ y
    )
    loss = 0.5 * np.mean((y - tx @ beta) ** 2)
    return beta, loss


# %% Logistic Regression
# Logistic regression using gradient descent (y ∈ {0, 1})
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Apply logistic regression using gradient descent.

    Args:
        y: shape=(N,)
        tx: shape=(N,D)
        initial_w: shape=(D,)
        max_iters: int
        gamma: float

    Returns:
        w: shape=(D,)
        loss: shape=(max_iters,)
    """
    w = initial_w
    loss = sigmoid_loss(y, tx, w)
    for _ in range(max_iters):
        gradient = sigmoid_derivative(y, tx, w)
        w = w - gamma * gradient
        loss = sigmoid_loss(y, tx, w)
    return w, loss


# %% Regularized Logistic Regression
# Regularized logistic regression using gradient descent (y ∈ {0, 1}, with regularization term λ∥w∥2)
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Apply regularized logistic regression using gradient descent.

    Args:
        y: shape=(N,)
        tx: shape=(N,D)
        lambda_: float
        initial_w: shape=(D,)
        max_iters: int
        gamma: float

    Returns:
        w: shape=(D,)
        loss: shape=(max_iters,)
    """
    w = initial_w
    loss = sigmoid_loss(y, tx, w)
    for n_iter in range(max_iters):
        gradient = sigmoid_derivative(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        print("Iter: {}".format(n_iter)," || ", "Loss:" , sigmoid_loss(y, tx, w))
        loss = sigmoid_loss(y, tx, w)
    return w, loss
