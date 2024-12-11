import openpyxl
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class MyBinaryClassifier:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.weights = None

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        if not X or not X[0]:  # Check if X is empty or contains empty rows
            raise ValueError("Training data is empty or malformed.")

        self.weights = [0.0] * (len(X[0]) + 1)  # Include bias weight
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                prediction = self._predict_row(xi)
                error = yi - prediction
                for j in range(len(xi)):
                    self.weights[j] += learning_rate * error * xi[j]
                self.weights[-1] += learning_rate * error  # Update bias weight

    def predict(self, X):
        return [1 if self._predict_row(row) >= self.threshold else 0 for row in X]

    def _predict_row(self, row):
        activation = sum(w * x for w, x in zip(self.weights[:-1], row)) + self.weights[-1]
        return 1 / (1 + math.exp(-activation))  # Sigmoid function

# Load and preprocess datasets from Excel
def read_excel(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    data = []
    for row in sheet.iter_rows(values_only=True):
        data.append(list(row))
    return data[1:]  # Skip the header

def normalize_units(row, indices):
    # Normalize specific weather attributes
    normalized_row = []
    for i in indices:
        value = row[i]
        if i in range(13, 16):  # Humidity (%) columns (max, avg, min)
            normalized_row.append(value / 100 if value is not None else None)
        elif i in range(19, 22):  # Wind Speed (mph) columns (max, avg, min)
            normalized_row.append(value)  # Leave wind speed as is
        else:
            normalized_row.append(value)  # For other attributes, use as is
    return normalized_row

def load_filtered_dataset(file_path):
    data = read_excel(file_path)
    filtered_data = []

    # Column indices for min, max, avg weather data (adjusted for zero-based indexing)
    min_max_avg_indices = list(range(10, 26))  # From column 11 (K) to column 26 (Z)

    for row in data:
        # Ensure all required columns have valid numerical data
        if all(isinstance(row[i], (int, float)) and row[i] is not None for i in min_max_avg_indices):
            normalized_row = normalize_units(row, min_max_avg_indices)
            filtered_data.append(normalized_row + [row[-1]])

    # Extract features (min, max, avg) and labels (UFO sightings)
    features = [row[:-1] for row in filtered_data]
    labels = [1 if row[-1] == 1 else 0 for row in filtered_data]  # Adjust index for label
    return features, labels

# Split data into training and testing sets
def split_data(X, y, test_ratio=0.2):
    combined = list(zip(X, y))
    random.shuffle(combined)
    split_idx = int(len(combined) * (1 - test_ratio))
    train_set, test_set = combined[:split_idx], combined[split_idx:]
    return (
        [x for x, _ in train_set], [y for _, y in train_set],
        [x for x, _ in test_set], [y for _, y in test_set]
    )

# Visualization

def visualize_feature_importance(weights, feature_names):  # Adjusted to include proper titles
    # Exclude the bias weight (last weight)
    feature_weights = weights[:-1]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_weights, color="skyblue")
    plt.xlabel("Weight Magnitude")
    plt.ylabel("Feature")
    plt.title("Feature Importance Based on Model Weights")
    plt.show()

def visualize_results(true_labels, predictions):
    tp = sum(1 for true, pred in zip(true_labels, predictions) if true == pred == 1)
    tn = sum(1 for true, pred in zip(true_labels, predictions) if true == pred == 0)
    fp = sum(1 for true, pred in zip(true_labels, predictions) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(true_labels, predictions) if true == pred == 0 and pred == 0)

    print(f"Confusion Matrix:")
    print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy: {accuracy:.2f}")

    # Pie chart visualization for better insight
    labels = ['True Positives', 'True Negatives', 'False Positives', 'False Negatives']
    values = [tp, tn, fp, fn]
    colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999']

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Classification Results')
    plt.show()

# Main script
if __name__ == "__main__":
    try:
        # Load and filter dataset
        features, labels = load_filtered_dataset('merged_weather_ufo.xlsx')

        # Debugging: Print dataset statistics
        print(f"Total samples: {len(features)}")
        print(f"Sample features: {features[:1]}")
        print(f"Sample labels: {labels[:1]}")

        # Split data
        X_train, y_train, X_test, y_test = split_data(features, labels)

        # Train binary classifier
        binary_classifier = MyBinaryClassifier()
        binary_classifier.fit(X_train, y_train)

        # Test binary classifier
        predictions = binary_classifier.predict(X_test)

        # Visualize results
        visualize_results(y_test, predictions)

        # Visualize feature importance
        feature_names = ["Temperature (°F) Max", "Temperature (°F) Avg", "Temperature (°F) Min", "Dew Point (°F) Max", "Dew Point (°F) Avg", "Dew Point (°F) Min", "Humidity (%) Max", "Humidity (%) Avg", "Humidity (%) Min", "Wind Speed (mph) Max", "Wind Speed (mph) Avg", "Wind Speed (mph) Min", "Pressure (in) Max", "Pressure (in) Avg", "Pressure (in) Min", "Precipitation (in) Total"]
        visualize_feature_importance(binary_classifier.weights, feature_names)

    except Exception as e:
        print(f"Error: {e}")
