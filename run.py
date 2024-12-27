import numpy as np
from preprocess import Preprocess
from KCV import *
from get_results import *

# Set a random seed to ensure reproducibility
np.random.seed(42)

K = 3
MAX_ITERS = 600
BEST_GAMMA = 1
BEST_LAMBDA = 1e-3


# Load the training and testing data from the .csv file
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset",sub_sample=False)

# Preprocess the data annd perform feature engineering
D = Preprocess(x_train, x_test, y_train, train_ids, test_ids)
D.process()

# Perform the K-Fold Cross Validation
P = K_Fold_CV(D.x_train, D.y_train, K)
P.train_and_evaluate(MAX_ITERS, BEST_GAMMA, BEST_LAMBDA)

mean_params = P.get_mean_parameters()

predictions = evaluate(mean_params, D.x_test)
create_csv_submission(test_ids, predictions, "submission.csv")