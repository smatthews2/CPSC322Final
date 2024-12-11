import openpyxl
import math
import random
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt

class MyRandomForestClassifier:
    def __init__(self, n_trees=20, m_trees=5, f_features=2):
        self.n_trees = n_trees
        self.m_trees = m_trees
        self.f_features = f_features
        self.forest = []

    def fit(self, X, y):
        # Step 1: Generate stratified test set
        test_size = len(X) // 3
        indices = list(range(len(X)))
        random.shuffle(indices)

        test_indices = indices[:test_size]
        remainder_indices = indices[test_size:]

        self.X_test = [X[i] for i in test_indices]
        self.y_test = [y[i] for i in test_indices]
        X_train = [X[i] for i in remainder_indices]
        y_train = [y[i] for i in remainder_indices]

        # Step 2: Generate N random decision trees
        trees = []
        for _ in range(self.n_trees):
            boot_indices = [random.choice(range(len(X_train))) for _ in range(len(X_train))]
            X_boot = [X_train[i] for i in boot_indices]
            y_boot = [y_train[i] for i in boot_indices]

            tree = MyDecisionTreeClassifier()
            tree.fit(X_boot, y_boot)
            trees.append(tree)

        # Step 3: Select M most accurate trees
        tree_accuracies = []
        for tree in trees:
            predictions = tree.predict(X_train)
            accuracy = sum(1 for pred, true in zip(predictions, y_train) if pred == true) / len(y_train)
            tree_accuracies.append((accuracy, tree))

        tree_accuracies.sort(reverse=True, key=lambda x: x[0])
        self.forest = [tree for _, tree in tree_accuracies[:self.m_trees]]

    def predict(self, X):
        votes = []
        for tree in self.forest:
            votes.append(tree.predict(X))

        majority_votes = []
        for i in range(len(X)):
            predictions = [vote[i] for vote in votes]
            majority_votes.append(max(set(predictions), key=predictions.count))

        return majority_votes

class MyDecisionTreeClassifier:
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return [self._classify(row, self.tree) for row in X]

    def _build_tree(self, X, y):
        if len(set(y)) == 1:
            return y[0]  # Leaf node with a single class

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return Counter(y).most_common(1)[0][0]  # Leaf node with majority class

        left_indices = [i for i, row in enumerate(X) if row[best_feature] <= best_threshold]
        right_indices = [i for i, row in enumerate(X) if row[best_feature] > best_threshold]

        left_tree = self._build_tree([X[i] for i in left_indices], [y[i] for i in left_indices])
        right_tree = self._build_tree([X[i] for i in right_indices], [y[i] for i in right_indices])

        return {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}

    def _best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature_index in range(len(X[0])):
            thresholds = set(row[feature_index] for row in X)
            for threshold in thresholds:
                left_indices = [i for i, row in enumerate(X) if row[feature_index] <= threshold]
                right_indices = [i for i, row in enumerate(X) if row[feature_index] > threshold]

                if not left_indices or not right_indices:
                    continue

                left_y = [y[i] for i in left_indices]
                right_y = [y[i] for i in right_indices]

                gain = self._information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent, left, right):
        p_entropy = self._entropy(parent)
        left_entropy = self._entropy(left)
        right_entropy = self._entropy(right)
        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)
        return p_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _entropy(self, y):
        counts = Counter(y)
        total = len(y)
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    def _classify(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        feature = tree["feature"]
        threshold = tree["threshold"]

        if row[feature] <= threshold:
            return self._classify(row, tree["left"])
        else:
            return self._classify(row, tree["right"])

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

        # Train and test MyRandomForestClassifier
        rf_classifier = MyRandomForestClassifier(n_trees=20, m_trees=5, f_features=2)
        rf_classifier.fit(X_train, y_train)
        rf_predictions = rf_classifier.predict(X_test)
        print("Random Forest Accuracy:", sum(1 for pred, true in zip(rf_predictions, y_test) if pred == true) / len(y_test))

        # Debugging: Print predictions and actual values
        print("Predictions vs Actuals:")
        for pred, actual in zip(rf_predictions, y_test):
            print(f"Predicted: {pred}, Actual: {actual}")

    except Exception as e:
        print(f"Error: {e}")
