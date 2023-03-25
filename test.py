import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load the data into a pandas DataFrame
data = pd.read_csv('data.csv')

print(data.keys())
# Convert the model_output and model_target values to binary
data[' model_output'] = (data[' model_output'] >= 0.5).astype(int)
data[' model_target'] = (data[' model_target'] >= 0.5).astype(int)

# Extract the predicted labels and the actual labels from the DataFrame
y_pred = data[' model_output']
y_true = data[' model_target']

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Print the confusion matrix
print(cm)