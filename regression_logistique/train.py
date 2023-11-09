import dataset as dt
import regression_logistique
import numpy as np
import csv
import sys

np.set_printoptions(threshold=sys.maxsize)

def load_dataset(file_path, tipo=0, balance=1.0):
    if tipo == 0:
        dataset = dt.TrainingDataset(file_path)
        dataset.balance_training_data(balance)
    else:
        dataset = dt.InferenceDataset(file_path)

    # dataset.balance_training_data()
    return dataset

def train_logistic_regression(dataset, n_iter=1000, lr=1e-6):
    logist = regression_logistique.LogisticRegression(dataset)
    logist.train(learning_rate=lr, n_iter=n_iter)
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
    train_data = load_dataset('../data/train.csv', balance=0.75)
    logistic_model = train_logistic_regression(train_data, n_iter=10000, lr=1e-6)

    inference = load_dataset('../data/test.csv', tipo=1)
    inference_data = inference.prepare_inference()

    accuracy_percentage, predicted_labels = evaluate_accuracy(logistic_model, train_data)

    print(f"Accuracy: {accuracy_percentage:.2f}%")
    print_class_distribution(train_data.train[1])

    write_predictions_to_csv_and_print_class_distribution(inference_data, logistic_model, "predictions_mock.csv")



