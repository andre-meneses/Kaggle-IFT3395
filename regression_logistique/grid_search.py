import dataset_refactor as dt
import regression_logistique
import numpy as np
import csv
import sys

np.set_printoptions(threshold=sys.maxsize)

# Functions defined previously (load_dataset, train_logistic_regression, evaluate_accuracy, etc.)

if __name__ == "__main__":
    learning_rates = [0.05, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7]
    iterations = [1000, 5000, 10000, 50000, 100000]
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

