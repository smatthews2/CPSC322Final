import openpyxl
import math
import random
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt

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
        else:
            normalized_row.append(value)  # For other attributes, use as is
    return normalized_row

def load_filtered_dataset(file_path):
    data = read_excel(file_path)
    filtered_data = []

    # Define relevant indices for min, max, avg weather data and label column
    relevant_indices = list(range(10, 26)) + [-1]  # Columns 11-26 + Label

    for row in data:
        # Eliminate rows with None in relevant columns
        if any(row[i] is None for i in relevant_indices):
            continue

        # Convert labels to numeric (1 for "yes", 0 for "no")
        label = row[-1]
        if isinstance(label, str):
            label = label.strip().lower()
            if label == "yes":
                label = 1
            elif label == "no":
                label = 0
            else:
                print(f"Row with invalid label: {row[-1]}")
                continue

        # Normalize and add valid rows
        if all(isinstance(row[i], (int, float)) for i in relevant_indices[:-1]):
            normalized_row = normalize_units(row, relevant_indices[:-1])
            filtered_data.append(normalized_row + [label])

    if not filtered_data:
        raise ValueError("No valid rows found in the dataset. Please check the data and column indices.")

    # Extract features and labels
    features = [row[:-1] for row in filtered_data]
    labels = [row[-1] for row in filtered_data]
    return features, labels

def split_data(X, y, test_ratio=0.2):
    combined = list(zip(X, y))
    random.shuffle(combined)
    split_idx = int(len(combined) * (1 - test_ratio))
    train_set, test_set = combined[:split_idx], combined[split_idx:]
    return (
        [x for x, _ in train_set], [y for _, y in train_set],
        [x for x, _ in test_set], [y for _, y in test_set]
    )

class MyKNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for test_point in X:
            neighbors = self._get_neighbors(test_point)
            prediction = self._majority_vote(neighbors)
            predictions.append(prediction)
        return predictions

    def _get_neighbors(self, test_point):
        distances = []
        for train_point, label in zip(self.X_train, self.y_train):
            distance = self._euclidean_distance(test_point, train_point)
            distances.append((distance, label))
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]

    def _euclidean_distance(self, point1, point2):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    def _majority_vote(self, neighbors):
        votes = Counter(label for _, label in neighbors)
        return votes.most_common(1)[0][0]

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

        # Train and test MyKNNClassifier
        knn_classifier = MyKNNClassifier(k=5)
        knn_classifier.fit(X_train, y_train)
        knn_predictions = knn_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = sum(1 for pred, true in zip(knn_predictions, y_test) if pred == true) / len(y_test)
        print("KNN Accuracy:", accuracy)

    except Exception as e:
        print(f"Error: {e}")
