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


    def balance_training_data(self, percent_most_common=0.7, seed=None):
        """
        Downsample the most common class in the training data by a specified percentage.

        Parameters:
            percent_most_common (float): The percentage of the most common class instances to keep.
            seed (int): Random seed for data shuffling.

        Returns:
            None (modifies the 'train' attribute in place).
        """
        # Calculate the counts for each class and find the most common class
        class_counts = np.bincount(self.train[1])
        most_common_class = np.argmax(class_counts)

        # Calculate the number of samples to keep for the most common class
        samples_to_keep = int(class_counts[most_common_class] * percent_most_common)

        # Get indices for the most common class
        most_common_indices = np.where(self.train[1] == most_common_class)[0]
        rng = np.random.default_rng(seed=seed)
        chosen_indices = rng.choice(most_common_indices, size=samples_to_keep, replace=False)

        # Get indices for the rest of the classes
        other_indices = np.where(self.train[1] != most_common_class)[0]

        # Combine indices and shuffle them
        balanced_indices = np.concatenate((chosen_indices, other_indices))
        balanced_indices = rng.permutation(balanced_indices)

        # Apply the new balanced indices to the training data
        self.train = (self.train[0][:, balanced_indices], self.train[1][balanced_indices])


    def _remove_duplicates(self, data):
        # print(data.shape)
        return np.unique(data, axis=0)

    def bin_and_combine_lat_lon(self,data, num_bins_lat, num_bins_lon):
        """
        Bin latitude and longitude data and combine into a single feature.

        Parameters:
        - data: A pandas DataFrame with 'latitude' and 'longitude' columns.
        - num_bins_lat: The number of bins for latitude.
        - num_bins_lon: The number of bins for longitude.

        Returns:
        - A pandas DataFrame with original data and additional columns for binned and combined features.
        """
        # Create bins for latitude and longitude
        data['lat_bin'] = pd.cut(data['lat'], bins=num_bins_lat, labels=range(num_bins_lat))
        data['lon_bin'] = pd.cut(data['lon'], bins=num_bins_lon, labels=range(num_bins_lon))

        data['lat_bin'] = data['lat_bin'].astype(int)
        data['lon_bin'] = data['lon_bin'].astype(int)

        # Combine the binned values into a single numeric feature
        data['location_combined'] = data['lat_bin'] * num_bins_lon + data['lon_bin']

        data.drop('lat', axis=1, inplace=True)
        data.drop('lon', axis=1, inplace=True)

        return data

    def _load_data(self, csv_path):
        """
        Loads the dataset from a CSV file and removes the first column (Sample Number).

        Args:
            csv_path (str): The file path to the CSV file.

        Returns:
            numpy.ndarray: The loaded dataset as a NumPy array.
        """
        df = pd.read_csv(csv_path)
        last_column = df.columns[-1]

        df = self.bin_and_combine_lat_lon(df,10,10)

        # Convert 'time' to string to ensure proper slicing
        df['time'] = df['time'].astype(str)

        # Extract year, month, and day into separate columns
        df['Year'] = df['time'].str[:4].astype(int)
        df['Month'] = df['time'].str[4:6].astype(int)
        df['Day'] = df['time'].str[6:].astype(int)

        # Adjust the year column to start from 0
        min_year = df['Year'].min()
        df['Year'] = df['Year'] - min_year

        # Drop the 'time' column
        df.drop('time', axis=1, inplace=True)

        # print(last_column)

        # Reorder the columns to put 'Year', 'Month', 'Day' before the last column
        df = df[[c for c in df if c not in ['Year', 'Month', 'Day', last_column]] + ['Year', 'Month', 'Day', last_column]]

        # Save the modified DataFrame to a CSV file
        df.to_csv('updated_file_teste.csv', index=False)

        data = df.to_numpy()

        # Remove the first column (Sample Number)
        data = data[:, 1:]
        # data = self._remove_duplicates(data)
        # print(data.shape)

        return data

    def prepare_inference(self):
        x = self.data[:,:-1].T
        # print(x.shape)
        # x = x / x.max(axis=0)
        x = np.log(x+1+abs(np.min(x)))
        homogeneous_coordinate = np.ones((1, x.shape[1]))

        return np.vstack((x, homogeneous_coordinate))

    def create_batches(self, batch_size):
        """
        Creates batches of data from the training set.

        Args:
            batch_size (int): The size of each batch.

        Returns:
            list: A list of batches, where each batch is a tuple containing (features, labels).
        """
        num_samples = self.train[0].shape[1]
        num_batches = num_samples // batch_size

        batches = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_features = self.train[0][:, start_idx:end_idx]
            batch_labels = self.train[1][start_idx:end_idx]
            batches.append((batch_features, batch_labels))

        # If there are remaining samples, create one more batch with the remaining samples
        if num_samples % batch_size != 0:
            start_idx = num_batches * batch_size
            batch_features = self.train[0][:, start_idx:]
            batch_labels = self.train[1][start_idx:]
            batches.append((batch_features, batch_labels))

        return batches

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

        data = self.data
        data = self._remove_duplicates(data)

        size = data.shape[0]
        split_size = int(train_split * size)

        rng = np.random.default_rng(seed=seed)
        random_idx = rng.permutation(size)

        train = data[random_idx[:split_size]]
        validation = data[random_idx[split_size:]]

        # x_train = train[:, :-1].T
        # x_val = validation[:, :-1].T
        x_train=train.T
        x_val=validation.T

        homogeneous_coordinate_train = np.ones((1, x_train.shape[1]))
        homogeneous_coordinate_val = np.ones((1, x_val.shape[1]))

        y_train = train[:, -1].astype(int)
        y_val = validation[:, -1].astype(int)

        return (np.vstack((x_train, homogeneous_coordinate_train)), y_train), (np.vstack((x_val, homogeneous_coordinate_val)), y_val)
        

if __name__=='__main__':

    filepath = "../data/train.csv"
    dataset = Dataset(filepath)

    # print(dataset.val[0].shape)
    # print(dataset.val[1].shape)
    # print(dataset.shape)
    # print(dataset.x)
    # print(dataset.y)
