import dataset as dt
import numpy as np
import csv
import regression_logistique
import pandas as pd


dataset = dt.Dataset('../data/train.csv')
logist = regression_logistique.LogisticRegression(*dataset.train)

# y = dataset.train[1]
# print(np.count_nonzero(y == 2)/len(y))
# print(y)
# print(len(y))


logist.train()

inference = dt.Dataset('../data/test.csv')
data = inference.prepare_inference()

csv_file = "predictions.csv"

val = dataset.val

correct_predictions = 0
total_samples = len(val[0].T)

y_correct = []
y_predicted = []

for i in range(len(val[0].T)):
    x = val[0].T[i]
    y = val[1][i]

    prediction = logist.predict(x)
    # print(y)
    # print(prediction)

    y_predicted.append(prediction)
    y_correct.append(y)

    # Assuming y_correct and y_predicted are lists
    if y == prediction:
        correct_predictions += 1

# print(y_predicted)
accuracy = correct_predictions / total_samples
accuracy_percentage = accuracy * 100

print(f"Accuracy: {accuracy_percentage:.2f}%")


with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["SNo", "Label"])

    for i, x in enumerate(data.T):
        writer.writerow([i,logist.predict(x)])
    
