import random
import math
from collections import Counter, defaultdict

class MyDecisionTreeClassifier:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        features = list(range(len(X[0])))
        if self.max_features:
            features = random.sample(features, self.max_features)
        self.tree = self._build_tree(X, y, features)

    def _build_tree(self, X, y, features):
        if len(set(y)) == 1:  # If all labels are the same, return the label
            return y[0]

        if not features or not X:
            return self._majority_label(y)

        best_feature = self._best_split_feature(X, y, features)
        tree = {best_feature: {}}

        feature_values = set(row[best_feature] for row in X)
        for value in feature_values:
            sub_X, sub_y = self._partition(X, y, best_feature, value)
            sub_features = [f for f in features if f != best_feature]
            tree[best_feature][value] = self._build_tree(sub_X, sub_y, sub_features)

        return tree

    def _partition(self, X, y, feature, value):
        X_subset, y_subset = [], []
        for i, row in enumerate(X):
            if row[feature] == value:
                X_subset.append(row)
                y_subset.append(y[i])
        return X_subset, y_subset

    def _best_split_feature(self, X, y, features):
        def entropy(labels):
            counts = Counter(labels)
            total = len(labels)
            return -sum((count / total) * math.log2(count / total) for count in counts.values())

        total_entropy = entropy(y)
        best_gain, best_feature = 0, None
        for feature in features:
            partitions = defaultdict(list)
            for i, row in enumerate(X):
                partitions[row[feature]].append(y[i])

            weighted_entropy = sum((len(partition) / len(y)) * entropy(partition) for partition in partitions.values())
            gain = total_entropy - weighted_entropy

            if gain > best_gain:
                best_gain, best_feature = gain, feature

        return best_feature

    def _majority_label(self, labels):
        return Counter(labels).most_common(1)[0][0]

    def predict(self, X):
        return [self._predict_single(row, self.tree) for row in X]

    def _predict_single(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        value = row[feature]
        if value not in tree[feature]:
            return None  # Handle missing splits
        return self._predict_single(row, tree[feature][value])


class MyRandomForestClassifier:
    def __init__(self, n_trees=10, max_features=None):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = MyDecisionTreeClassifier(max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]
        return X_sample, y_sample

    def predict(self, X):
        tree_predictions = [tree.predict(X) for tree in self.trees]
        return [self._majority_vote(predictions) for predictions in zip(*tree_predictions)]

    def _majority_vote(self, predictions):
        return Counter(predictions).most_common(1)[0][0]


# Load and preprocess the weather dataset
def load_weather_data(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip().split(',') for line in file]
    
    header = data[0]
    rows = data[1:]
    
    # Extract features and labels
    features = [[float(val) if val.replace('.', '', 1).isdigit() else val for val in row[:-1]] for row in rows]
    labels = [row[-1] for row in rows]
    return features, labels

# Split a dataset
def split_data(X, y, test_ratio=0.2):
    combined = list(zip(X, y))
    random.shuffle(combined)
    split_idx = int(len(combined) * (1 - test_ratio))
    train_set, test_set = combined[:split_idx], combined[split_idx:]
    return (
        [x for x, _ in train_set], [y for _, y in train_set],
        [x for x, _ in test_set], [y for _, y in test_set]
    )

# Example usage
X, y = load_weather_data('weather_data_combined2.csv')
X_train, y_train, X_test, y_test = split_data(X, y)

rf = MyRandomForestClassifier(n_trees=5, max_features=3)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

print(f"Predictions: {predictions}")
