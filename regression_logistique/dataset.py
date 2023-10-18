import numpy as np
import pandas as pd

class Dataset:
    """
    A class for handling datasets, with loading, splitting, and encoding functionalities.
    """

    def __init__(self, csv_path, train_split=0.8, seed=None):
        """
        Initializes the Dataset object.

        Args:
            csv_path (str): The file path to the CSV file containing the dataset.
            train_split (float): The proportion of data to be used for training (default is 0.8).
            seed (int): Random seed for data shuffling (default is None).

        Attributes:
            data (numpy.ndarray): The loaded dataset.
            n_classes (int): Number of classes in the dataset.
            train (tuple): A tuple containing the training data (features, labels).
            val (tuple): A tuple containing the validation data (features, labels).
            one_hot_labels (numpy.ndarray): One-hot encoded labels for training data.
        """
        self.data = self._load_data(csv_path)
        self.n_classes = 3
        self.train, self.val = self._split_dataset(train_split, seed)
        self.one_hot_labels = self._one_hot_encoding()

    def _one_hot_encoding(self):
        """
        One-hot encodes the labels for the training data.

        Returns:
            numpy.ndarray: A one-hot encoded array of shape (number of training samples, n_classes).
        """
        one_hot_labels = np.zeros((self.train[1].shape[0], self.n_classes))
        one_hot_labels[np.arange(self.train[1].shape[0]), self.train[1]] = 1
        return one_hot_labels

    def _load_data(self, csv_path):
        """
        Loads the dataset from a CSV file and removes the first column (Sample Number).

        Args:
            csv_path (str): The file path to the CSV file.

        Returns:
            numpy.ndarray: The loaded dataset as a NumPy array.
        """
        data = pd.read_csv(csv_path)
        data = data.to_numpy()

        # Remove the first column (Sample Number)
        data = data[:, 1:]

        return data

    def _split_dataset(self, train_split, seed):
        """
        Splits the dataset into training and validation sets.

        Args:
            train_split (float): The proportion of data to be used for training.
            seed (int): Random seed for data shuffling.

        Returns:
            tuple: A tuple containing training and validation data as follows:
                (x_train, y_train): Features and labels for training.
                (x_val, y_val): Features and labels for validation.
        """
        size = self.data.shape[0]
        split_size = int(train_split * size)

        rng = np.random.default_rng(seed=seed)
        random_idx = rng.permutation(size)

        train = self.data[random_idx[:split_size]]
        validation = self.data[random_idx[split_size:]]

        x_train = train[:, :-1]
        x_val = validation[:, :-1]

        y_train = train[:, -1].astype(int)
        y_val = validation[:, -1].astype(int)

        return (x_train, y_train), (x_val, y_val)

if __name__=='__main__':

    filepath = "../data/train.csv"
    dataset = Dataset(filepath)

    print(dataset.one_hot_labels)

    # print(dataset.val[0].shape)
    # print(dataset.val[1].shape)
    # print(dataset.x)
    # print(dataset.y)
