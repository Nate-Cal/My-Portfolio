import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# TODO: CHANGE URL TO LOCATION OF WINE.DATA FILE
# Load the dataset 
url = 'wine.data'
data = pd.read_csv(url)

# Separate features and target
X = data.drop('Class', axis=1)
Y = data['Class']

# 70% train, 15% dev, 15% test
# Split off training set, then split remaining into development and test
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y)
# Split the temp set (30%) into development and test 
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nTraining set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"Development set size: {len(X_dev)} ({len(X_dev)/len(X)*100:.1f}%)")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)
X_test_scaled = scaler.transform(X_test)

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def train(self, X, y):
        # Only need to store training data, no actual training
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances to all training samples using Euclidean distance 
            # sqrt(sum((x1 - x2)^2))
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Get indices of k nearest neighbors; not really sure what [:self.k] does
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k nearest neighbors; not really sure what .iloc does
            k_nearest_labels = self.y_train.iloc[k_indices]
            
            # Vote: most common label
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)

# Tune k value
print("\n--- Tuning k value ---")
k_values = [1, 3, 5, 7, 9]
k_scores = []

for k in k_values:
    knn = KNN(k=k)
    knn.train(X_train_scaled, y_train)
    knn_pred = knn.predict(X_dev_scaled)
    acc = accuracy_score(y_dev, knn_pred)
    k_scores.append(acc)
    print(f"k={k:2d}: Accuracy = {acc:.3f}")

# Best k
best_k = k_values[np.argmax(k_scores)]
best_knn_acc = max(k_scores)
print(f"\nBest k value: {best_k}")
print(f"Best KNN accuracy on dev set: {best_knn_acc:.3f}")

# Train final KNN with best k
knn_best = KNN(k=best_k)
knn_best.train(X_train_scaled, y_train)
knn_best_pred = knn_best.predict(X_dev_scaled)


# Evaluate KNN on test set
knn_test_pred = knn_best.predict(X_test_scaled)
knn_test_acc = accuracy_score(y_test, knn_test_pred)
