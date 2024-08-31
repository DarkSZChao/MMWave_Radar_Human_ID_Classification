import json
import os
import pickle
import time
from collections import deque, Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import load_model

from NN import OrthogonalRegularizer
from config import *


# load data and label
dataset_file = 'test_dataset_12_200_filtered_timelen_30'
with open(f'{os.path.join(DATASET_DIR, dataset_file)}', 'rb') as file:
    test_data_np, test_labels_np = pickle.load(file)

# Load your trained Keras model
model = load_model(os.path.join(BESTMODEL_SAVE_DIR, '200_filtered_timelen_30_tnet/30.h5'), custom_objects={'OrthogonalRegularizer': OrthogonalRegularizer})


# # Generate predictions
# predictions = model.predict(test_data_np)
# # Convert predictions to class labels
# pred_labels = np.argmax(predictions, axis=1)

# blur predict
blur_len = 60
pred_labels_total = np.array([])
for person_label in range(12):
    person_data = test_data_np[test_labels_np == person_label]

    # evaluate the model
    possibility = model.predict(person_data)
    pred_labels = np.argmax(possibility, axis=1)

    pred_labels_blur = np.empty(pred_labels.shape, dtype=np.int8)
    pred_labels_blur_deque = deque([], blur_len)
    for idx, label in enumerate(pred_labels):
        pred_labels_blur_deque.append(label)
        if pred_labels_blur_deque.__len__() == pred_labels_blur_deque.maxlen:
            label, _ = Counter(pred_labels_blur_deque).most_common(1)[0]
            pred_labels_blur[idx] = label
        else:
            pred_labels_blur[idx] = label
    pred_labels = pred_labels_blur
    pred_labels_total = np.concatenate([pred_labels_total, pred_labels])

# Create confusion matrix
conf_matrix = confusion_matrix(test_labels_np, pred_labels_total)

# Flatten confusion matrix for sklearn's functions
y_true = []
y_pred = []
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        y_true.extend([i + 1] * conf_matrix[i, j])
        y_pred.extend([j + 1] * conf_matrix[i, j])
# Calculate precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# draw confusion matrix
conf_matrix = (conf_matrix.T / conf_matrix.sum(axis=1)).T

# # Display the confusion matrix
# disp = ConfusionMatrixDisplay(conf_matrix, display_labels=list(range(12)))
# disp.plot(cmap=plt.cm.Blues, values_format='.1%')
#
# plt.show()


classes = list(range(12))

plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

for i in range(len(classes)):
    for j in range(len(classes)):
        cell_value = conf_matrix[i, j]
        cell_text = "{:.1%}".format(cell_value / np.sum(conf_matrix[i, :]))
        plt.text(j, i, cell_text, horizontalalignment='center', color='white' if cell_value > np.max(conf_matrix) / 2 else 'black')

plt.tight_layout()
plt.show()
