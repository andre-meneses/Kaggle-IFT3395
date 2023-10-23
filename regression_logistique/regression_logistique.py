import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, x_train, y_train, variance=1, regularization_parameter=5):
        """
        Class constructor.

        Parameters:
            x_train (numpy.ndarray): Input feature matrix.
            y_train (numpy.ndarray): True class labels.
            variance (float): Variance for weight initialization.
        """
        self.weights = self._initialize_weights(variance)
        self.x = x_train
        self.y = y_train
        self.regularization_parameter = regularization_parameter

    def _cross_entropy_loss(self, probs):
        return np.sum(-np.log(probs[self.y]))

    def _softmax(self, x):
        """
        Compute the softmax function for the given input.

        Parameters:
            x (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Softmax probabilities.
        """
        # print(f"x: {x}")
        # print(f"weights: {self.weights.T}")
        # print(f"product: {self.weights.T @ x}")
        logits = np.exp(self.weights.T @ x)
        return logits / np.sum(logits)


    def _delta(self, i, j):
        "Compute the kronecker delta for a given i and j"

        return 1 if i == j else 0

    def _initialize_weights(self, variance):
        """
        Initialize the model weights.

        Parameters:
            dim (int): Dimension of the weight vector.
            variance (float): Variance for weight initialization.

        Returns:
            numpy.ndarray: Initialized weight vector.
        """
        mean = 0
        return np.random.normal(mean, np.sqrt(variance), (19,3))

    def _jacobian(self, X, Y):
        """
        Compute the Jacobian matrix for a softmax regression model.

        Parameters:
            X (numpy.ndarray): Input feature matrix.
            Y (numpy.ndarray): True class labels.

        Returns:
            numpy.ndarray: The Jacobian matrix.

        """
        gradient = np.zeros(self.weights.T.shape)

        for x,y in zip(X.T, Y):
            S = self._softmax(x)

            for i in range(gradient.shape[0]):
                delta = self._delta(y,i)

                for j in range(gradient.shape[1]):
                    gradient[i,j] += (S[i] - delta)*x[j]

        # Regularization term
        gradient += 2 * self.regularization_parameter * self.weights.T

        # print(gradient)
        return gradient


    def train(self, learning_rate=0.001, n_iter=10000, verbose=True):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
            learning_rate (float): Learning rate for gradient descent.
            n_iter (int): Number of iterations for training.
        """

        loss = []

        for i in range(n_iter):
            self.weights -= learning_rate * self._jacobian(self.x, self.y).T

            if verbose:
                print(f"Iteration n: {i}")

    def predict(self, x):
        """
        Predict the class label for the input.

        Parameters:
            x (numpy.ndarray): Input feature vector.

        Returns:
            int: Predicted class label.
        """
        # print(x)
        # print(self.weights.T @ x)
        return np.argmax(self.weights.T @ x)

