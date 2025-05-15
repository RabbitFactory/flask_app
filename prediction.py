# Importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Loading and reading the dataset
heart = pd.read_csv("heart_cleveland_upload.csv")

# Creating a copy of dataset so that it will not affect our original dataset
heart_df = heart.copy()

# Renaming the columns 
heart_df = heart_df.rename(columns={'condition':'target'})

# Model building 
# Fixing our data in x and y. Here y contains target data and X contains all other features
X = heart_df.drop(columns='target')
y = heart_df.target

# Splitting our dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating Random Forest classifier
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Creating a pickle file for the classifier
# Note: Despite the filename suggesting KNN, this is actually a RandomForest model
filename = 'heart-disease-prediction-knn-model.pkl'
pickle.dump(model, open(filename, 'wb'))

print(f'\nModel saved as {filename}')
