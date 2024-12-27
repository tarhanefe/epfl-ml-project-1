import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from helpers import *
from utils import *

class K_Fold_CV:

    def __init__(self, X_train, y_train, k):
        """
        Initializes the K-Fold Cross-Validation class.

        Args:
            X_train (numpy array): Training features.
            y_train (numpy array): Training labels (binary classification).
            k (int): Number of folds for cross-validation.
            model_function (function): The logistic regression model function to be used
                                        (e.g., `logistic_regression` or `reg_logistic_regression`).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.model_function = reg_logistic_regression

        # Split data into k folds
        self.X_train_folds = np.array_split(self.X_train, self.k)
        self.y_train_folds = np.array_split(self.y_train, self.k)

        # To store metrics and parameters for each fold
        self.train_accuracies = []
        self.test_accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1s = []
        self.train_losses = []
        self.test_losses = []
        self.trained_parameters = []

    def train_and_evaluate(self, max_iters=1000, gamma=0.01, lambda_=0.1):
        """
        Trains and evaluates the model using k-fold cross-validation with undersampling.
        """
        
        self.hyperparameters = {"max_iters": max_iters, "gamma": gamma, "lambda": lambda_}
        self.train_losses = []  # Reset to store the entire loss log for each fold

        for i in range(self.k):
            # Prepare the validation data for this fold
            X_val = self.X_train_folds[i]
            y_val = self.y_train_folds[i]
            X_train_full = np.concatenate(
                [self.X_train_folds[j] for j in range(self.k) if j != i]
            )
            y_train_full = np.concatenate(
                [self.y_train_folds[j] for j in range(self.k) if j != i]
            )

            # Initialize weights for this fold
            initial_w = np.random.randn(X_train_full.shape[1])

            # Train the model once using the undersampled data and reg_logistic_regression function
            w, train_loss_log = self.model_function(
                y_train_full, X_train_full, lambda_, initial_w, max_iters, gamma
            )

            # Store the training loss for this fold
            self.train_losses.append(train_loss_log)

            # Evaluate on the validation data for the current fold
            train_accuracy = self.compute_accuracy(X_train_full, y_train_full, w)
            self.train_accuracies.append(train_accuracy)
            
            metrics = self.calculate_metrics(y_val, X_val @ w)
            
            self.precisions.append(metrics["precision"])
            self.recalls.append(metrics["recall"])
            self.f1s.append(metrics["f1_score"])

            val_loss = self.evaluate_loss(y_val, X_val, w)
            val_accuracy = self.compute_accuracy(X_val, y_val, w)
            self.test_losses.append(val_loss)
            self.test_accuracies.append(val_accuracy)

            # Store trained parameters for this fold
            self.trained_parameters.append(w)

    def evaluate_loss(self, y, X, w):
        """Computes cross-entropy loss for logistic regression."""
        print(np.isnan(X).any())
        pred = sigmoid(X @ w)
        
        loss = -np.mean(y * np.log(pred + 1e-10) + (1 - y) * np.log(1 - pred + 1e-10))
        return loss

    def compute_accuracy(self, X, y, w):
        """Computes accuracy for binary classification tasks."""
        predictions = sigmoid(X @ w) >= 0.5
        correct_predictions = np.sum(predictions == y)
        return correct_predictions / len(y) * 100

    def get_metrics(self):
        """
        Returns the collected metrics and trained parameters.
        """
        return {
            "train_accuracies": self.train_accuracies,
            "test_accuracies": self.test_accuracies,
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "trained_parameters": self.trained_parameters,
        }

    def plot_metrics(self):
        """
        Plots the training and validation accuracies and losses across the k-folds.
        """
        # Plot accuracies
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(
            range(1, self.k + 1),
            self.train_accuracies,
            label="Train Accuracy",
            marker="o",
        )
        plt.plot(
            range(1, self.k + 1),
            self.test_accuracies,
            label="Validation Accuracy",
            marker="x",
        )
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.title("Train vs Validation Accuracy Across Folds")
        plt.legend()

        # Plot losses
        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, self.k + 1), self.train_losses, label="Train Loss", marker="o"
        )
        plt.plot(
            range(1, self.k + 1), self.test_losses, label="Validation Loss", marker="x"
        )
        plt.xlabel("Fold")
        plt.ylabel("Loss")
        plt.title("Train vs Validation Loss Across Folds")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_loss_decrease(self):
        """
        Plots the decrease of the loss across iterations for each fold.
        """
        plt.figure(figsize=(10, 6))

        for i, train_loss_log in enumerate(self.train_losses):
            plt.plot(train_loss_log, label=f"Fold {i+1}")

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Decrease Across Iterations for Each Fold")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_best_parameters(self):
        """
        Returns the parameters from the fold that had the highest validation accuracy.
        """
        best_index = np.argmax(self.test_accuracies)
        return self.trained_parameters[best_index], self.test_accuracies[best_index]

    def get_mean_parameters(self):
        """
        Returns the mean of the trained parameters across all folds.
        """
        return np.mean(self.trained_parameters, axis=0)

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculates accuracy, precision, recall, and F1 score for binary classification.

        Args:
            y_true (numpy array): True binary labels.
            y_pred (numpy array): Predicted binary labels.

        Returns:
            dict: A dictionary containing accuracy, precision, recall, and F1 score.
        """
        # Convert predictions to binary format (0 or 1)
        y_pred = (y_pred >= 0.5).astype(int)

        # Calculate True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((y_pred == 1) & (y_true == 1))  # True Positives
        fp = np.sum((y_pred == 1) & (y_true == 0))  # False Positives
        tn = np.sum((y_pred == 0) & (y_true == 0))  # True Negatives
        fn = np.sum((y_pred == 0) & (y_true == 1))  # False Negatives

        # Calculate accuracy
        accuracy = (tp + tn) / len(y_true) * 100  # Convert to percentage

        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Print and return metrics
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def write_mean_metrics(self):
        self.best_params = self.get_mean_parameters()
        
        with open("metrics.txt", "a") as file:
            file.write(f"Max Iterations: {self.hyperparameters['max_iters']}    Learning Rate (Gamma): {self.hyperparameters['gamma']}    Lambda: {self.hyperparameters['lambda']}\n \
                        Accuracy: {np.mean(self.test_accuracies)} +- {np.std(self.test_accuracies)}\n \
                        Precision: {np.mean(self.precisions)} +- {np.std(self.precisions)}\n \
                        Recall: {np.mean(self.recalls)} +- {np.std(self.recalls)}\n \
                        F1 Score: {np.mean(self.f1s)} +- {np.std(self.f1s)}\n \
                        ************************************************************\n")