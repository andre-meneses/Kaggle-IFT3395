import dataset_refactor as dt
import regression_logistique
import numpy as np
import csv
import sys

np.set_printoptions(threshold=sys.maxsize)

def load_dataset(file_path, tipo=0):
    if tipo == 0:
        dataset = dt.TrainingDataset(file_path)
        dataset.balance_training_data(1.0)
    else:
        dataset = dt.InferenceDataset(file_path)

    # dataset.balance_training_data()
    return dataset

def train_logistic_regression(dataset, n_iter=1000, lr=1e-6):
    logist = regression_logistique.LogisticRegression(dataset)
    logist.train(learning_rate=lr, n_iter=n_iter, verbose=False)
    # print(logist.weights)
    return logist

def evaluate_accuracy(logist, dataset):
    val = dataset.val
    correct_predictions = 0

    y_correct = []
    y_predicted = []

    for i in range(len(val[0].T)):
        x = val[0].T[i]
        y = val[1][i]

        prediction = logist.predict(x)
        y_predicted.append(prediction)
        y_correct.append(y)

        if y == prediction:
            correct_predictions += 1

    total_samples = len(val[0].T)
    accuracy = correct_predictions / total_samples
    accuracy_percentage = accuracy * 100

    # print(y_predicted)

    return accuracy_percentage, y_predicted

def write_predictions_to_csv(data, logist, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNo", "Label"])

        for i, x in enumerate(data.T):
            writer.writerow([i + 1, logist.predict(x)])

def print_class_distribution(y_train):
    class_counts = np.bincount(y_train)
    for class_label, count in enumerate(class_counts):
        print(f"Class {class_label}: {count} samples")

def write_predictions_to_csv_and_print_class_distribution(data, logist, csv_file):
    class_distribution = {}

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNo", "Label"])

        for i, x in enumerate(data.T):
            # print(x.shape)
            prediction = logist.predict(x)
            writer.writerow([i + 1, prediction])

            # Count the predicted classes
            if prediction in class_distribution:
                class_distribution[prediction] += 1
            else:
                class_distribution[prediction] = 1

    # Print the inferred class distribution
    print("Inferred class distribution:")
    for class_label, count in class_distribution.items():
        print(f"Class {class_label}: {count} samples")


if __name__ == "__main__":
    learning_rates = [1e-5, 1e-6, 1e-7]
    iterations = [1000, 5000, 10000]
    balance_percents = [1.0, 0.75, 0.5]

    best_accuracy = 0
    best_params = {'lr': None, 'n_iter': None, 'balance_percent': None}
    best_model = None

    with open('grid_search_results.csv', 'w', newline='') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(['Learning Rate', 'Iterations', 'Balance Percentage', 'Accuracy'])

        for lr in learning_rates:
            for n_iter in iterations:
                for balance_percent in balance_percents:
                    # Load and balance the training dataset based on the current balance_percent
                    train_data = load_dataset('../data/train.csv')
                    train_data.balance_training_data(balance_percent)

                    # Train the logistic regression model with the current parameters
                    logistic_model = train_logistic_regression(train_data, n_iter=n_iter, lr=lr)

                    # Evaluate the model's accuracy on the validation set
                    accuracy_percentage, _ = evaluate_accuracy(logistic_model, train_data)

                    # Write the current grid search results to the CSV file
                    writer.writerow([lr, n_iter, balance_percent, accuracy_percentage])

                    print(f"Tested: LR={lr}, Iter={n_iter}, Balance={balance_percent}, Accuracy={accuracy_percentage:.2f}%")

                    # Update the best model if the current model has the highest accuracy so far
                    if accuracy_percentage > best_accuracy:
                        best_accuracy = accuracy_percentage
                        best_params = {'lr': lr, 'n_iter': n_iter, 'balance_percent': balance_percent}
                        best_model = logistic_model

    # Print the best model's parameters and its accuracy
    print(f"Best model parameters: {best_params}")
    print(f"Best model accuracy: {best_accuracy:.2f}%")

    # Load the test dataset and evaluate the best model on it
    test_data = load_dataset('../data/test.csv', tipo=1)
    test_inference_data = test_data.prepare_inference()

    # You can now write the predictions to a CSV file or perform further analysis
    write_predictions_to_csv_and_print_class_distribution(test_inference_data, best_model, "best_model_predictions.csv")

