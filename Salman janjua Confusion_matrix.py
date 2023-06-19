import numpy as np
from sklearn.metrics import confusion_matrix

# Define the actual labels and predicted labels
actual_labels = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1, 1])
predicted_labels = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 1])

# Create the confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)

# Extract the values from the confusion matrix
true_negative = cm[0, 0]
false_positive = cm[0, 1]
false_negative = cm[1, 0]
true_positive = cm[1, 1]

# Calculate accuracy
accuracy = (true_positive + true_negative) / np.sum(cm)

# Calculate precision
precision = true_positive / (true_positive + false_positive)

# Calculate recall
recall = true_positive / (true_positive + false_negative)

# Calculate F1 score
f1_score = 2 * (precision * recall) / (precision + recall)

# Calculate Type I error
type_i_error = false_positive / (false_positive + true_negative)

# Calculate Type II error
type_ii_error = false_negative / (false_negative + true_positive)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Type I Error:", type_i_error)
print("Type II Error:", type_ii_error)