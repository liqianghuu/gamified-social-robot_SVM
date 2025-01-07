import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Read data from CSV
df = pd.read_csv('generated_dataset.csv')

# Encode binary values (0 for change, 1 for same)
df['Label'] = df['Label'].map({'up': 1, 'down': 0, 'same': 2})

# Split the data into features and target
X = df[['ReactionTime', 'Engagement', 'Accuracy']]
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train an SVM model
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)