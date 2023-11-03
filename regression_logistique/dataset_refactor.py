import numpy as np
import pandas as pd

class BaseDataset:
    """
    A base class for handling datasets, with loading and encoding functionalities.
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = self._load_data(csv_path)
        self.n_classes = 3  # Assuming this is known and fixed for all datasets

    def _load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        last_column = df.columns[-1]
        df = self._transform_geographical_features(df)
        df = self._transform_time_feature(df, last_column)
        self.data = df.to_numpy()[:,1:]
        self.normalized_data = self._normalize_features(df).to_numpy()[:,1:]
        return self.normalized_data

    def _normalize_features(self, data):
        """
        Normalize the specified features with min-max scaling and the rest with z-score normalization.

        Parameters:
        - data: A pandas DataFrame containing the dataset.

        Returns:
        - The DataFrame with normalized features.
        """
        label_column_name = 'Label'  # replace with your actual label column name
        if label_column_name in data.columns:
            label_column = data[label_column_name]
            data = data.drop(label_column_name, axis=1)
        else:
            label_column = None

        # Separate columns for different normalization
        min_max_columns = ['Year', 'Month', 'Day', 'location_combined']
        z_score_columns = [col for col in data.columns if col not in min_max_columns]

        # Apply Min-Max normalization to specified columns
        for col in min_max_columns:
            if col in data.columns:
                col_data = data[col].values.astype(float)
                min_val = np.min(col_data)
                max_val = np.max(col_data)
                # Avoid division by zero
                range_val = max_val - min_val if max_val != min_val else 1
                data[col] = (col_data - min_val) / range_val

        # Apply Z-Score normalization to the rest
        for col in z_score_columns:
            col_data = data[col].values.astype(float)
            mean = np.mean(col_data)
            std = np.std(col_data)
            # Avoid division by zero
            std = std if std != 0 else 1
            data[col] = (col_data - mean) / std

        # If label_column is present, concatenate it back to the dataframe
        if label_column is not None:
            data = pd.concat([data, label_column], axis=1)

        self._save_normalized_data_to_csv(data)

        return data

        
    def _save_normalized_data_to_csv(self,data):
        data.to_csv('normalized_data.csv', index=False)

    def _transform_geographical_features(self, df, num_bins_lat=10, num_bins_lon=10):
        # Implement the geographical feature transformation logic
        df['lat_bin'] = pd.cut(df['lat'], bins=num_bins_lat, labels=range(num_bins_lat))
        df['lon_bin'] = pd.cut(df['lon'], bins=num_bins_lon, labels=range(num_bins_lon))

        df['lat_bin'] = df['lat_bin'].astype(int)
        df['lon_bin'] = df['lon_bin'].astype(int)

        # Combine the binned values into a single numeric feature
        df['location_combined'] = df['lat_bin'] * num_bins_lon + df['lon_bin']

        df.drop('lat', axis=1, inplace=True)
        df.drop('lat_bin', axis=1, inplace=True)
        df.drop('lon', axis=1, inplace=True)
        df.drop('lon_bin', axis=1, inplace=True)

        return df

    def _transform_time_feature(self, df, last_column):
        # Implement the time feature transformation logic
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

        # Reorder the columns to put 'Year', 'Month', 'Day' before the last column
        df = df[[c for c in df if c not in ['Year', 'Month', 'Day', last_column]] + ['Year', 'Month', 'Day', last_column]]

        # Save the modified DataFrame to a CSV file
        df.to_csv('updated_file_teste.csv', index=False)

        return df

    def _remove_duplicates(self, data):
        return np.unique(data, axis=0)

    # ... (any other common methods)


class TrainingDataset(BaseDataset):
    """
    A subclass for handling training datasets, including data splitting and balancing.
    """

    def __init__(self, csv_path, train_split=0.8, seed=None):
        super().__init__(csv_path)
        self.train_split = train_split
        self.seed = seed
        self.train, self.val = self._split_dataset()

    def _split_dataset(self):
        data = self._remove_duplicates(self.normalized_data)

        size = data.shape[0]
        split_size = int(self.train_split * size)

        rng = np.random.default_rng(seed=self.seed)
        random_idx = rng.permutation(size)

        train = data[random_idx[:split_size]]
        validation = data[random_idx[split_size:]]

        x_train = train[:, :-1].T
        x_val = validation[:, :-1].T

        homogeneous_coordinate_train = np.ones((1, x_train.shape[1]))
        homogeneous_coordinate_val = np.ones((1, x_val.shape[1]))

        y_train = train[:, -1].astype(int)
        y_val = validation[:, -1].astype(int)

        return (np.vstack((x_train, homogeneous_coordinate_train)), y_train), (np.vstack((x_val, homogeneous_coordinate_val)), y_val)

    def balance_training_data(self):
        # Implement the balancing logic
        pass

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


    # ... (any other training-specific methods)


class InferenceDataset(BaseDataset):
    """
    A subclass for handling inference datasets.
    """

    def __init__(self, csv_path):
        super().__init__(csv_path)

    def prepare_inference_data(self):
        x = self.normalized_data.T
        homogeneous_coordinate = np.ones((1, x.shape[1]))
        return np.vstack((x, homogeneous_coordinate))

    # ... (any other inference-specific methods)


if __name__ == '__main__':
    # Example usage for training data
    training_filepath = "../data/train.csv"
    training_dataset = TrainingDataset(training_filepath)

    # Example usage for inference data
    # inference_filepath = "../data/test.csv"  # Assuming test.csv is the inference dataset without labels
    # inference_dataset = InferenceDataset(inference_filepath)
    # inference_data = inference_dataset.prepare_inference_data()
