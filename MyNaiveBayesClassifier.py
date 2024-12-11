import openpyxl
import math
import random
from collections import Counter, defaultdict
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

class MyNaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = defaultdict(float)
        self.feature_probabilities = defaultdict(lambda: defaultdict(float))

    def fit(self, X, y):
        # Calculate class probabilities
        total_samples = len(y)
        class_counts = Counter(y)
        for label, count in class_counts.items():
            self.class_probabilities[label] = count / total_samples

        # Calculate feature probabilities
        num_features = len(X[0])
        for label in class_counts:
            subset = [X[i] for i in range(total_samples) if y[i] == label]
            for feature_index in range(num_features):
                feature_values = [row[feature_index] for row in subset]
                mean = sum(feature_values) / len(feature_values)
                variance = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
                self.feature_probabilities[label][feature_index] = (mean, variance)

    def predict(self, X):
        predictions = []
        for row in X:
            label_probabilities = {}
            for label, class_prob in self.class_probabilities.items():
                label_probabilities[label] = math.log(class_prob)
                for feature_index, feature_value in enumerate(row):
                    mean, variance = self.feature_probabilities[label][feature_index]
                    if variance == 0:  # Avoid division by zero
                        variance = 1e-6
                    probability = self._gaussian_pdf(feature_value, mean, variance)
                    label_probabilities[label] += math.log(probability)
            predictions.append(max(label_probabilities, key=label_probabilities.get))
        return predictions

    def _gaussian_pdf(self, x, mean, variance):
        return (1 / math.sqrt(2 * math.pi * variance)) * math.exp(-((x - mean) ** 2) / (2 * variance))

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

        # Train and test MyNaiveBayesClassifier
        nb_classifier = MyNaiveBayesClassifier()
        nb_classifier.fit(X_train, y_train)
        nb_predictions = nb_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = sum(1 for pred, true in zip(nb_predictions, y_test) if pred == true) / len(y_test)
        print("Naive Bayes Accuracy:", accuracy)

        # Debugging: Print predictions and actual values
        print("Predictions vs Actuals:")
        for pred, actual in zip(nb_predictions, y_test):
            print(f"Predicted: {pred}, Actual: {actual}")

        # Debugging: Print class distributions
        print(f"Training set class distribution: {Counter(y_train)}")
        print(f"Test set class distribution: {Counter(y_test)}")

    except Exception as e:
        print(f"Error: {e}")
