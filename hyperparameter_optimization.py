import numpy as np 
from helpers import *
from preprocess import Preprocess
from KCV import * 
from implementations import *

# Set a random seed to ensure reproducibility
np.random.seed(42)

# Initiates a K-Fold cross validation with the given hyperparameter values with the regularized logistic regression model and reports its metrics
def run_network(x_train, x_test, y_train, train_ids, test_ids, max_iters, gamma, lambda_, K):
    D = Preprocess(x_train, x_test, y_train, train_ids, test_ids)
    D.process()
    P = K_Fold_CV(D.x_train, D.y_train, K)
    P.train_and_evaluate(max_iters, gamma, lambda_)
    P.write_mean_metrics()

K = 3
max_iters = 600
gammas = [1e-3,1e-2,1e-1,1]
lambdas = [1e-4,1e-3,1e-2,1e-1]

# Load the training and testing data from the .csv file
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset",sub_sample=False)

# Initiate hyperparameter optimization using grid search on different gamma and lambda values
# The best performing model for the task at hand is found to be regularized logistic regression 
cnt = 0
for gamma in gammas:
    for lambda_ in lambdas:
        cnt += 1
        print("Count:", cnt,"Gamma: ", gamma, "Lambda: ", lambda_)
        run_network(x_train, x_test, y_train, train_ids, test_ids, max_iters, gamma, lambda_, K)
        