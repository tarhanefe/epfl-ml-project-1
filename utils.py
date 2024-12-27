import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """apply sigmoid function on x.

    Args:
        x: shape=(N, 1)

    Returns:
        a vector of shape (N, 1)
    """
    return np.exp(x) / (1 + np.exp(x))


def sigmoid_derivative(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    gradient = tx.T @ (sigmoid(tx @ w) - y) / len(y)
    return gradient


def sigmoid_loss(y, tx, w):
    """
    Compute the logistic loss for binary labels 0 and 1.

    Args:
        y: shape=(N, 1), binary labels (0, 1)
        tx: shape=(N, D), features
        w: shape=(D, 1), weights

    Returns:
        A scalar loss value
    """
    # Compute cross-entropy loss
    loss = np.mean(-y * (tx @ w) + np.log(1 + np.exp(tx @ w)))
    return loss
