# The file for getting the final results for submission for logistic regression
import numpy as np
from utils import *
from helpers import *


def evaluate(params, X_test):
    """Evaluate the model on the test data. Using sigmoid function and parameters from logistic regression

    Args:
        params: shape=(D,)
        X_test: shape=(N,D)

    Returns:
        y_pred: shape=(N,)
    """
    y_pred = np.round(sigmoid(X_test @ params))
    y_pred[y_pred == 0] = -1
    return y_pred
