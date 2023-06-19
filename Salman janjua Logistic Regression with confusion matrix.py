import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Reshaped for Logistic function.
X = np.array([1.38, 7.74, 2.09, 0.24, 1.42, 1.45, 8.92, 4.47, 4.16, 4.52, 2.69, 5.88]).reshape(-1, 1)
y = np.array([0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1])

logr = linear_model.LogisticRegression()
logr.fit(X, y)

# Predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(np.array([3.46]).reshape(-1, 1))
print("Predicted:", predicted)

# Compute the confusion matrix
y_pred = logr.predict(X)
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate and print the accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Calculate and print the precision
precision = precision_score(y, y_pred)
print("Precision:", precision)

# Calculate and print the recall
recall = recall_score(y, y_pred)
print("Recall:", recall)

# Calculate and print the F1 score
f1 = f1_score(y, y_pred)
print("F1 Score:", f1)
