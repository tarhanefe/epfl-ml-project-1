# CS-433 Machine Learning Project 1 

This README.md file serves as an instructional document for the codebase that is created for the first project of the course CS-433 Machine Learning of EPFL for the Fall 2024 semester.

## Placement of the Data Folder

In order to reproduce the results presented in the report, make sure that the files ```x_train.csv```, ```x_test.csv```, and ```y_train.csv``` are placed under a folder named ```dataset``` inside the project repository for the data loading process to run successfully.

## Breakdown of the Files Inside the Codebase

Here is a quick breakdown of the files present inside the codebase. Each file within the repository will be listed along with their purpose and function respectively.

```bash
implementations.py
```
This file consists of the implementations of six different machine learning methods that have been mentioned in the project description document.

```bash
utils.py
```
utils.py includes some utility functions that are needed by the methods in implementations.py for certain computations.

```bash
helpers.py
```
The exact same file featuring some useful helper functions with the one already provided to us with the project announcement.

```bash
preprocess.py
```
This file is the class description file of the Python class ```Preprocess```, which is simply used to clean and to preprocess the project dataset by utilizing several strategies, which are meticulously explained inside the method docstrings of the class. To name a few, it handles both continous and categorical features, removes unrelated columns from the dataset, and replaces ```NaN``` values by sampling values from a Gaussian distribution with mean and standard deviation of the column having ```NaN``` values.

Before starting a training or a hyperparameter optimization run, a Preprocess object is created just after loading training and testing data. Then, the dataset is analyzed and feature engineering is performed via the method ```Preprocess.process()```.

```bash
KCV.py
```
This file includes the class description of the Python class ```K_Fold_CV```, which is simply used to perform a K-Fold Cross Validation on the training set with some additional options.

Since we found out that regularized logistic regression is the top-performing model among the other simple models implemented in ```implementations.py```, ```K_Fold_CV``` uses regularized logistic regression by default.

To create an instance of ```K_Fold_CV```, one needs to pass the preprocessed versions of ```x_train``` and ```y_train```, and also the hyperparameter value ```k```. After an instance of the class is created, one can simply call the ```K_Fold_CV.train_and_evaluate()``` method to initiate a cross validation, and call ```K_Fold_CV.write_mean_metrics()``` to write out the training results to a file named ```metrics.txt```.

```bash
hyperparameter_optimization.py
```
As can be inferred from the name, this file is used to start a hyperparameter optimization run with grid search for our best-performing model, regularized logistic regression. The hyperparameter values used for the search can be seen inside the file:

```python
K = 3
max_iters = 600
gammas = [1e-3,1e-2,1e-1,1]
lambdas = [1e-4,1e-3,1e-2,1e-1]
```

To obtain the results presented in the project report regarding the hyperparameter search procedure, one can simply run this file with ```python hyperparameter_optimization.py```. Note that it is assumed the files ```x_train.csv```, ```x_test.csv```, and ```y_train.csv``` are placed under a folder named ```dataset``` inside the project repository for the data loading process to run successfully. After the run terminates, a file named ```metrics.txt``` will be created within the project scope and it will contain the cross-validation metrics for different hyperparameter combinations.

```bash
run.py
```
This is the file responsible for generating the file ```submission.csv``` which consists of the predictions of our best performing model with hyperparameters discovered via a hyperparameter optimization run. One can simply run the file without changing anything, and obtain the predictions that are submitted by us to the aicrowd.com online competition system.