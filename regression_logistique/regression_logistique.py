import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, dataset, variance=1, regularization_parameter=5):
        """
        Class constructor.

        Parameters:
            x_train (numpy.ndarray): Input feature matrix.
            y_train (numpy.ndarray): True class labels.
            variance (float): Variance for weight initialization.
        """
        self.weights_shape = (dataset.data.shape[1],3)
        self.weights = self._initialize_weights(variance)
        self.dataset = dataset
        self.regularization_parameter = regularization_parameter

    def _cross_entropy_loss(self, probs, labels):
        return np.sum(-np.log(probs[labels]))

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
        # print(x.shape)
        # print(self.weights.T.shape)
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
        # return np.random.normal(mean, np.sqrt(2), self.weights_shape)
        return np.random.uniform(0,1,self.weights_shape)
        # return np.zeros(self.weights_shape)

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
        # gradient *= np.array([0.8, 1.5, 1.1]).reshape(-1,1)

        # print(gradient)
        return gradient

    def train(self, learning_rate=1e-6, n_iter=10000, verbose=True, batch_size=256):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
            dataset (Dataset): The dataset object containing training data.
            learning_rate (float): Learning rate for gradient descent.
            n_iter (int): Number of iterations for training.
        """
        history_loss = []

        for i in range(n_iter):
            batches = self.dataset.create_batches(batch_size)  # Create batches using the dataset object

            loss = []

            for batch_features, batch_labels in batches:
                loss.append(self._cross_entropy_loss(self._softmax(batch_features), batch_labels))
                self.weights -= learning_rate * self._jacobian(batch_features, batch_labels).T

            history_loss.append(mean(loss))

            if verbose:
                print(f"Iteration n: {i}")


        plt.title("Perte Moyenne x Nombre d'Iterations")
        plt.ylabel("Perte moyenne par batch")
        plt.xlabel("Nombre d'Iterations")
        plt.plot(range(n_iter), history_loss)
        plt.show()

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
        # print(x.shape)
        # print(self.weights.T.shape)
        return np.argmax(self.weights.T @ x)

